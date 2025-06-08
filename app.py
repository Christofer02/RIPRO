import streamlit as st
import os
import re
import numpy as np
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from nltk import download
from docx import Document
from functools import reduce
import spacy
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import FOAF, XSD
from rank_bm25 import BM25Okapi
import json

# Download NLTK stopwords
download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))
english_stopwords = set(stopwords.words('english'))
stopwords_set = spanish_stopwords | english_stopwords

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    st.error("Please install the spaCy English model by running: python -m spacy download en_core_web_md")
    st.stop()

# Define RDF namespaces
EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")

# Utility functions for text file processing
def load_text(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read().decode('utf-8') if isinstance(uploaded_file.read(), bytes) else uploaded_file.read()
        return content
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return '\n'.join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Tipo de archivo no soportado")

def normalize_text(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

def clean_words(text):
    tokens = tokenize(normalize_text(text))
    return [word for word in tokens if word not in stopwords_set and word.isalpha()]

def cosine_similarity(vec1, vec2):
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vectorize(words, vocab):
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)

def highlight_terms(text, query):
    terms = set(clean_words(query))
    for term in sorted(terms, key=len, reverse=True):
        if term:
            pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
            text = pattern.sub(r'<mark>\1</mark>', text)
    return text

# Functions for CSV processing
def extract_dates_prices_models(df):
    """Extract dates, prices, and product models from DataFrame"""
    # Dates from reviews.date
    dates = df['reviews.date'].dropna().tolist()
    
    # Prices from prices column (JSON-like) and reviews.text
    prices = []
    for price_data in df['prices'].dropna():
        try:
            price_list = json.loads(price_data.replace("'", "\""))
            for item in price_list:
                amount = item.get('amountMax', '')
                currency = item.get('currency', '')
                if amount and currency:
                    prices.append(f"{amount} {currency}")
        except json.JSONDecodeError:
            continue
    price_pattern = r'\b(price|cost)\s*[:\s]*\$?\s*(\d+(?:\.\d{2})?)\b'
    for review in df['reviews.text'].dropna():
        found_prices = re.findall(price_pattern, review, re.IGNORECASE)
        prices.extend([f"{p[1]} USD" for p in found_prices])
    
    # Models from manufacturerNumber and reviews.text
    models = df['manufacturerNumber'].dropna().unique().tolist()
    model_pattern = r'\b(model|modelo)\s*[:\s]*([A-Za-z0-9\-]+)\b'
    for review in df['reviews.text'].dropna():
        found_models = re.findall(model_pattern, review, re.IGNORECASE)
        models.extend([m[1] for m in found_models])
    
    return {
        "dates": list(set(dates)),  # Remove duplicates
        "prices": list(set(prices)),
        "models": list(set(models))
    }

def extract_entities(text):
    """Extract entities using spaCy NER"""
    doc = nlp(text)
    entities = {"Producto": [], "Marca": [], "Lugar": [], "Persona": []}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["Marca"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["Lugar"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["Persona"].append(ent.text)
    # Custom rule for products (nouns not tagged as other entities)
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in stopwords_set and token.text not in entities["Marca"] + entities["Lugar"] + entities["Persona"]:
            entities["Producto"].append(token.text)
    return entities

def extract_events(text):
    """Extract events like (Usuario, bought, Producto) or (Producto, failed, Fecha)"""
    doc = nlp(text)
    events = []
    for sent in doc.sents:
        verbs = [token for token in sent if token.pos_ == "VERB"]
        for verb in verbs:
            if verb.lemma_.lower() in ["buy", "purchase", "order"]:
                subj = next((t for t in verb.children if t.dep_ == "nsubj"), None)
                obj = next((t for t in verb.children if t.dep_ == "dobj"), None)
                if subj and obj:
                    events.append((subj.text, verb.lemma_, obj.text))
            elif verb.lemma_.lower() in ["fail", "malfunction", "break", "drain"]:
                subj = next((t for t in verb.children if t.dep_ == "nsubj"), None)
                date = next((t.text for t in sent if t.ent_type_ == "DATE"), None)
                if subj and date:
                    events.append((subj.text, verb.lemma_, date))
    return events

def extract_relations(text):
    """Extract relations like (Usuario, recommended, Producto) or (Marca, launched, Producto)"""
    doc = nlp(text)
    relations = []
    for sent in doc.sents:
        for verb in [t for t in sent if t.pos_ == "VERB"]:
            if verb.lemma_.lower() in ["recommend", "love", "enjoy"]:
                subj = next((t for t in verb.children if t.dep_ == "nsubj"), None)
                obj = next((t for t in verb.children if t.dep_ == "dobj"), None)
                if subj and obj:
                    relations.append((subj.text, verb.lemma_, obj.text))
            elif verb.lemma_.lower() in ["launch", "release"]:
                subj = next((t for t in verb.children if t.dep_ == "nsubj"), None)
                obj = next((t for t in verb.children if t.dep_ == "dobj"), None)
                if subj and obj:
                    relations.append((subj.text, verb.lemma_, obj.text))
    return relations

def build_rdf_graph(df):
    """Build RDF graph from DataFrame"""
    g = Graph()
    for idx, row in df.iterrows():
        user = URIRef(EX + row.get("reviews.username", f"user_{idx}").replace(" ", "_").replace(".", ""))
        product = URIRef(EX + row.get("name", f"product_{idx}").replace(" ", "_").replace("&", "and"))
        review = str(row.get("reviews.text", ""))
        
        # Add entities
        g.add((user, RDF.type, FOAF.Person))
        g.add((product, RDF.type, SCHEMA.Product))
        g.add((product, SCHEMA.name, Literal(row.get("name", ""))))
        g.add((product, SCHEMA.brand, Literal(row.get("brand", ""))))
        
        # Add events
        events = extract_events(review)
        for subj, verb, obj in events:
            if verb.lower() in ["buy", "purchase", "order"]:
                g.add((user, EX.bought, product))
                g.add((product, EX.boughtBy, user))
            elif verb.lower() in ["fail", "malfunction", "break", "drain"]:
                g.add((product, EX.failedOn, Literal(obj, datatype=XSD.date)))
        
        # Add relations
        relations = extract_relations(review)
        for subj, verb, obj in relations:
            if verb.lower() in ["recommend", "love", "enjoy"]:
                g.add((user, EX.recommended, product))
            elif verb.lower() in ["launch", "release"]:
                brand_uri = URIRef(EX + row.get("brand", "brand_" + str(idx)).replace(" ", "_"))
                g.add((brand_uri, EX.launched, product))
        
        # Add sentiment based on rating
        rating = row.get("reviews.rating", 0)
        sentiment = "positivo" if rating >= 4 else "negativo"
        g.add((product, EX.hasSentiment, Literal(sentiment)))
    
    return g

def rank_reviews_bm25(reviews, query):
    """Rank reviews using BM25"""
    tokenized_reviews = [clean_words(str(review)) for review in reviews]
    bm25 = BM25Okapi(tokenized_reviews)
    tokenized_query = clean_words(query)
    scores = bm25.get_scores(tokenized_query)
    return scores

# Streamlit app
st.sidebar.title("Buscador de documentos y análisis de CSV")
st.sidebar.header("Grupo 7", divider='rainbow')
st.sidebar.markdown("Soria, C., Zurita, M.")
st.sidebar.markdown("Modelo algebraico y NLP")

st.header('Selecciona tus archivos', divider='rainbow')

# File uploaders
uploaded_files = st.file_uploader(
    "Arrastra y suelta hasta 20 archivos (.txt, .docx)",
    type=["txt", "docx"],
    accept_multiple_files=True
)
csv_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

# Process text files
if uploaded_files:
    if len(uploaded_files) > 20:
        st.warning("Por favor, sube un máximo de 20 archivos.")
    else:
        raw_docs = list(map(load_text, uploaded_files))
        st.subheader("Paso 1: Carga de documentos")
        st.write("Documentos cargados en `raw_docs`.")
        
        st.subheader("Paso 2: Limpieza y tokenización")
        search_query = st.text_input("Escribe el texto que deseas buscar")
        if search_query:
            clean_docs = list(map(clean_words, raw_docs))
            clean_query = clean_words(search_query)
            
            st.subheader("Paso 3: Vocabulario único")
            vocab = sorted(set(reduce(lambda x, y: x + y, clean_docs, clean_query)))
            
            st.subheader("Paso 4: Vectorización")
            doc_vectors = list(map(lambda doc: vectorize(doc, vocab), clean_docs))
            query_vector = vectorize(clean_query, vocab)
            
            st.subheader("Paso 5: Similitud coseno")
            scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors]
            ranked = sorted(
                [(file, doc, score) for file, doc, score in zip(uploaded_files, raw_docs, scores) if score > 0],
                key=lambda x: x[2], reverse=True
            )
            
            if ranked:
                for rank, (file, doc, score) in enumerate(ranked, 1):
                    st.write(f"{rank}. {os.path.basename(file.name)} - Similitud coseno: {score:.4f}")
                    highlighted_doc = highlight_terms(doc, search_query)
                    with st.expander(f"Ver contenido del documento"):
                        st.markdown(
                            f'<div style="max-height: 200px; overflow-y: auto; background: #f9f9f9; padding: 8px; border-radius: 4px; white-space: pre-wrap;">{highlighted_doc}</div>',
                            unsafe_allow_html=True
                        )

# Process CSV file
if csv_file:
    st.subheader("Análisis de CSV")
    try:
        df = pd.read_csv(csv_file)
        st.write("CSV cargado. Primeras filas:")
        st.dataframe(df.head())
        
        # 2. Limpieza y Extracción con Regex
        st.subheader("2. Limpieza y Extracción con Regex")
        extracted = extract_dates_prices_models(df)
        st.write("Fechas extraídas:", extracted["dates"])
        st.write("Precios extraídos:", extracted["prices"])
        st.write("Modelos extraídos:", extracted["models"])
        
        # 3. NER
        st.subheader("3. Extracción de Entidades (NER)")
        review_text = " ".join(df['reviews.text'].dropna().astype(str))
        entities = extract_entities(review_text)
        entities["Producto"].extend(df['name'].dropna().unique().tolist())
        entities["Marca"].extend(df['brand'].dropna().unique().tolist())
        entities["Producto"] = list(set(entities["Producto"]))
        entities["Marca"] = list(set(entities["Marca"]))
        st.write("Entidades detectadas:", entities)
        
        # 4. Extracción de Eventos
        st.subheader("4. Extracción de Eventos")
        events = extract_events(review_text)
        st.write("Eventos extraídos:", events)
        
        # 5. Extracción de Relaciones
        st.subheader("5. Extracción de Relaciones")
        relations = extract_relations(review_text)
        st.write("Relaciones extraídas:", relations)
        
        # 6. Representación del Conocimiento (RDF)
        st.subheader("6. Representación del Conocimiento (RDF)")
        rdf_graph = build_rdf_graph(df)
        st.write("Grafo RDF creado. Ejemplo de tripletas:")
        for s, p, o in list(rdf_graph)[:10]:
            st.write(f"({s}, {p}, {o})")
        
        # 7. Web Semántica y Búsqueda
        st.subheader("7. Web Semántica y Búsqueda")
        query = st.text_input("Consulta para buscar reseñas similares (BM25)")
        if query:
            reviews = df['reviews.text'].dropna().tolist()
            scores = rank_reviews_bm25(reviews, query)
            ranked_reviews = sorted(zip(reviews, scores), key=lambda x: x[1], reverse=True)[:5]
            st.write("Reseñas más relevantes:")
            for rank, (review, score) in enumerate(ranked_reviews, 1):
                st.write(f"{rank}. Score: {score:.4f} - {review[:100]}...")
    except Exception as e:
        st.error(f"Error al procesar el CSV: {str(e)}")

if not uploaded_files and not csv_file:
    st.info("Por favor, sube archivos de texto (.txt, .docx) o un archivo CSV para comenzar el análisis.")