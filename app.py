import streamlit as st
import os
import re
import numpy as np
import pandas as pd
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk import download
import spacy
from rank_bm25 import BM25Okapi
from rdflib import Graph, RDF, RDFS, Literal, URIRef
from functools import reduce
from collections import defaultdict
import json
import plotly.express as px
import pickle
import hashlib
from pathlib import Path

# Configuración de cache para Streamlit
st.set_page_config(page_title="Semantic Search Engine", layout="wide")

# Descarga de recursos NLTK (solo una vez)
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    return True

# Cargar modelo de spaCy (solo una vez)
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please install the spaCy model by running: python -m spacy download en_core_web_sm")
        st.stop()

# Configurar stopwords (solo una vez)
@st.cache_resource
def setup_stopwords():
    download_nltk_resources()
    spanish_stopwords = set(stopwords.words('spanish'))
    english_stopwords = set(stopwords.words('english'))
    return spanish_stopwords | english_stopwords

# Inicializar recursos
nlp = load_spacy_model()
stopwords_set = setup_stopwords()

# Función para generar hash del archivo
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

# Cache para datos procesados
@st.cache_data
def load_and_process_csv_data(file_content, file_hash):
    """Carga y procesa completamente el CSV con cache basado en hash del archivo"""
    
    # Crear DataFrame desde el contenido del archivo
    from io import StringIO
    df = pd.read_csv(StringIO(file_content.decode('utf-8')))
    df['text_content'] = df['reviews.text'].fillna('') + ' ' + df['reviews.title'].fillna('')
    
    # Limitar el dataset para pruebas (opcional)
    # df = df.head(1000)  # Procesar solo las primeras 1000 filas para testing
    
    raw_docs = df['text_content'].tolist()
    doc_names = [f"Review_{i+1}" for i in range(len(raw_docs))]
    
    # Procesar todos los pasos de una vez
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 2: Cleaning and Regex
    status_text.text("Procesando: Limpieza de texto y extracción regex...")
    cleaned_docs = []
    prices_list = []
    models_list = []
    
    for i, doc in enumerate(raw_docs):
        cleaned, prices, models = clean_text(doc)
        cleaned_docs.append(cleaned)
        prices_list.append(prices)
        models_list.append(models)
        
        if i % 100 == 0:  # Actualizar progreso cada 100 documentos
            progress_bar.progress((i + 1) / len(raw_docs) * 0.2)
    
    tokenized_docs = [clean_words(doc) for doc in cleaned_docs]
    progress_bar.progress(0.2)
    
    # Step 3: NER (optimizado por lotes)
    status_text.text("Procesando: Reconocimiento de entidades nombradas...")
    entities_list = []
    batch_size = 50  # Procesar en lotes para optimizar spaCy
    
    for i in range(0, len(raw_docs), batch_size):
        batch = raw_docs[i:i+batch_size]
        batch_entities = [extract_entities(doc) for doc in batch]
        entities_list.extend(batch_entities)
        progress_bar.progress(0.2 + (i / len(raw_docs)) * 0.2)
    
    # Step 4: Event Extraction
    status_text.text("Procesando: Extracción de eventos...")
    events_list = []
    for i, doc in enumerate(raw_docs):
        events_list.append(extract_events(doc))
        if i % 100 == 0:
            progress_bar.progress(0.4 + (i / len(raw_docs)) * 0.2)
    
    # Step 5: Relation Extraction
    status_text.text("Procesando: Extracción de relaciones...")
    relations_list = []
    for i, (doc, entities) in enumerate(zip(raw_docs, entities_list)):
        relations_list.append(extract_relations(doc, entities))
        if i % 100 == 0:
            progress_bar.progress(0.6 + (i / len(raw_docs)) * 0.2)
    
    # Step 6: Emotion Extraction
    status_text.text("Procesando: Extracción de emociones...")
    emotions_list = []
    for i, (row, doc) in enumerate(zip(df.iterrows(), raw_docs)):
        emotions_list.append(extract_emotions(doc, row[1]['reviews.rating']))
        if i % 100 == 0:
            progress_bar.progress(0.8 + (i / len(raw_docs)) * 0.1)
    
    # Step 7: RDF Creation
    status_text.text("Procesando: Creación de grafo RDF...")
    rdf_graph = create_rdf_triplets(df, entities_list, events_list, relations_list, emotions_list)
    progress_bar.progress(0.95)
    
    # Preparar BM25
    status_text.text("Finalizando: Preparando índice de búsqueda...")
    vocab = sorted(set(reduce(lambda x, y: x + y, tokenized_docs, [])))
    progress_bar.progress(1.0)
    
    status_text.text("¡Procesamiento completado!")
    
    return {
        'df': df,
        'raw_docs': raw_docs,
        'doc_names': doc_names,
        'cleaned_docs': cleaned_docs,
        'tokenized_docs': tokenized_docs,
        'prices_list': prices_list,
        'models_list': models_list,
        'entities_list': entities_list,
        'events_list': events_list,
        'relations_list': relations_list,
        'emotions_list': emotions_list,
        'rdf_graph': rdf_graph,
        'vocab': vocab
    }

# Funciones optimizadas para procesamiento
def clean_text(text):
    """Clean and normalize text, extract prices and models with regex"""
    if not isinstance(text, str):
        return "", [], []
    
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    
    price_pattern = r'\b(?:\$|€|CAD|USD|EUR)?\s?\d+(?:\.\d{2})?\b'
    prices = re.findall(price_pattern, text)
    
    model_pattern = r'\b(?:kindle\s\w+|fire\s\w+|alexa\s\w+|paperwhite|voyage)\b'
    models = re.findall(model_pattern, text, re.IGNORECASE)
    
    tokens = re.findall(r'\b\w+\b', text)
    cleaned_tokens = [word for word in tokens if word not in stopwords_set and word.isalpha()]
    
    return ' '.join(cleaned_tokens), prices, models

def extract_entities(text):
    """Extract entities using spaCy"""
    if len(text) > 1000000:  # Limitar texto muy largo
        text = text[:1000000]
        
    doc = nlp(text)
    entities = {
        'PRODUCT': [],
        'BRAND': [],
        'LOCATION': [],
        'PERSON': []
    }
    for ent in doc.ents:
        if ent.label_ == 'PRODUCT':
            entities['PRODUCT'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['BRAND'].append(ent.text)
        elif ent.label_ == 'GPE':
            entities['LOCATION'].append(ent.text)
        elif ent.label_ == 'PERSON':
            entities['PERSON'].append(ent.text)
    return entities

def extract_events(text):
    """Extract events based on key verbs"""
    if len(text) > 1000000:  # Limitar texto muy largo
        text = text[:1000000]
        
    event_verbs = ['compré', 'bought', 'devolví', 'returned', 'probé', 'tried', 'funcionó', 'worked', 'falló', 'failed']
    events = []
    doc = nlp(text)
    
    for token in doc:
        if token.lemma_.lower() in event_verbs:
            for chunk in doc.noun_chunks:
                if chunk.root.head == token:
                    event = {
                        'verb': token.lemma_.lower(),
                        'subject': None,
                        'object': chunk.text
                    }
                    for child in token.children:
                        if child.dep_ == 'nsubj':
                            event['subject'] = child.text
                    events.append(event)
    
    return events

def extract_relations(text, entities):
    """Extract relations between entities"""
    relations = []
    doc = nlp(text[:1000000])  # Limitar texto
    
    for sent in doc.sents:
        if 'recommend' in sent.text.lower():
            for person in entities.get('PERSON', []):
                for product in entities.get('PRODUCT', []):
                    relations.append((person, 'recommended', product))
        if 'launch' in sent.text.lower():
            for brand in entities.get('BRAND', []):
                for product in entities.get('PRODUCT', []):
                    relations.append((brand, 'launched', product))
    
    return relations

def extract_emotions(text, rating):
    """Extract emotions based on sentiment and keywords"""
    emotion_dict = {
        'positive': ['happy', 'great', 'awesome', 'love', 'excellent', 'fantastic'],
        'negative': ['disappointed', 'bad', 'poor', 'hate', 'terrible', 'frustrating'],
        'neutral': ['okay', 'average', 'fine']
    }
    emotions = []
    text = text.lower()
    
    if rating >= 4:
        emotions.append('positive')
    elif rating <= 2:
        emotions.append('negative')
    else:
        emotions.append('neutral')
    
    for emotion, keywords in emotion_dict.items():
        if any(keyword in text for keyword in keywords):
            emotions.append(emotion)
    
    return list(set(emotions))

def sanitize_uri_component(text):
    """Sanitize text to create valid URI components."""
    text = re.sub(r'[^\w\s-]', '', text)
    text = text.replace(' ', '_').strip('_')
    return text

def create_rdf_triplets(df, entities_list, events_list, relations_list, emotions_list):
    """Create RDF graph from entities, events, relations, and emotions."""
    g = Graph()
    namespace = "http://example.org/amazon_reviews#"
    
    for idx, row in df.iterrows():
        review_id = URIRef(f"{namespace}review_{idx}")
        product = row['name']
        sanitized_product = sanitize_uri_component(product)
        product_uri = URIRef(f"{namespace}product_{sanitized_product}")
        
        g.add((product_uri, RDF.type, URIRef(f"{namespace}Product")))
        g.add((product_uri, RDFS.label, Literal(product)))
        
        for entity_type, entities in entities_list[idx].items():
            for entity in entities:
                sanitized_entity = sanitize_uri_component(entity)
                entity_uri = URIRef(f"{namespace}{entity_type.lower()}_{sanitized_entity}")
                g.add((entity_uri, RDF.type, URIRef(f"{namespace}{entity_type}")))
                g.add((entity_uri, RDFS.label, Literal(entity)))
                g.add((review_id, URIRef(f"{namespace}mentions"), entity_uri))
        
        for event in events_list[idx]:
            event_uri = URIRef(f"{namespace}event_{idx}_{event['verb']}")
            g.add((event_uri, RDF.type, URIRef(f"{namespace}Event")))
            g.add((event_uri, URIRef(f"{namespace}verb"), Literal(event['verb'])))
            if event['subject']:
                sanitized_subject = sanitize_uri_component(event['subject'])
                subject_uri = URIRef(f"{namespace}person_{sanitized_subject}")
                g.add((event_uri, URIRef(f"{namespace}subject"), subject_uri))
            sanitized_object = sanitize_uri_component(event['object'])
            g.add((event_uri, URIRef(f"{namespace}object"), URIRef(f"{namespace}product_{sanitized_object}")))
        
        for relation in relations_list[idx]:
            subject, predicate, obj = relation
            sanitized_subject = sanitize_uri_component(subject)
            sanitized_obj = sanitize_uri_component(obj)
            subject_uri = URIRef(f"{namespace}{sanitized_subject}")
            obj_uri = URIRef(f"{namespace}{sanitized_obj}")
            g.add((subject_uri, URIRef(f"{namespace}{predicate}"), obj_uri))
        
        for emotion in emotions_list[idx]:
            emotion_uri = URIRef(f"{namespace}emotion_{emotion}")
            g.add((emotion_uri, RDF.type, URIRef(f"{namespace}Emotion")))
            g.add((emotion_uri, RDFS.label, Literal(emotion)))
            g.add((product_uri, URIRef(f"{namespace}has_emotion"), emotion_uri))
        
        sentiment = 'positive' if row['reviews.rating'] >= 4 else 'negative' if row['reviews.rating'] <= 2 else 'neutral'
        g.add((product_uri, URIRef(f"{namespace}has_sentiment"), Literal(sentiment)))
    
    return g

# Funciones de búsqueda y visualización (mantener las originales)
@st.cache_data
def perform_search(query, tokenized_docs, doc_names, raw_docs, emotions_list, _vocab):
    """Realizar búsqueda con cache"""
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = clean_words(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(
        [(name, doc, score) for name, doc, score in zip(doc_names, tokenized_docs, scores) if score > 0],
        key=lambda x: x[2], reverse=True
    )
    
    # Cosine Similarity
    doc_vectors = [vectorize(doc, _vocab) for doc in tokenized_docs]
    query_vector = vectorize(tokenized_query, _vocab)
    cosine_scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors]
    
    # Combine scores
    combined_ranked = []
    for i, (name, doc, bm25_score) in enumerate(ranked):
        cosine_score = cosine_scores[doc_names.index(name)]
        combined_score = 0.7 * bm25_score + 0.3 * cosine_score
        review_idx = doc_names.index(name)
        combined_ranked.append((name, raw_docs[review_idx], combined_score, emotions_list[review_idx]))
    
    return sorted(combined_ranked, key=lambda x: x[2], reverse=True)[:10]

# Resto de funciones helper (mantener las originales)
def tokenize(text):
    return re.findall(r'\b\w+\b', text)

def clean_words(text):
    tokens = tokenize(normalize_text(text))
    return [word for word in tokens if word not in stopwords_set and word.isalpha()]

def normalize_text(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

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

def create_entity_type_bar_chart(rdf_graph, ranked_reviews):
    """Create entity type frequency chart"""
    namespace = "http://example.org/amazon_reviews#"
    entity_counts = defaultdict(int)
    
    for s, p, o in rdf_graph:
        if str(p).endswith('type'):
            entity_type = str(o).split('#')[-1]
            entity_counts[entity_type] += 1
    
    entity_types = list(entity_counts.keys())
    counts = [entity_counts[et] for et in entity_types]
    
    df_plot = pd.DataFrame({
        'Entity Type': entity_types,
        'Total Count': counts
    })
    
    fig = px.bar(
        df_plot,
        x='Entity Type',
        y='Total Count',
        title='Entity Type Frequency in Reviews'
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

def create_product_emotion_bar_chart(df, emotions_list, ranked_reviews):
    """Create product emotion distribution chart"""
    top_reviews = {name for name, _, _, _ in ranked_reviews[:10]}
    emotion_counts = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        if f"Review_{idx+1}" in top_reviews:
            product = row['name']
            for emotion in emotions_list[idx]:
                emotion_counts[product][emotion] += 1
    
    products = []
    emotions = []
    counts = []
    for product, emo_dict in emotion_counts.items():
        for emotion, count in emo_dict.items():
            products.append(product)
            emotions.append(emotion)
            counts.append(count)
    
    df_plot = pd.DataFrame({
        'Product': products,
        'Emotion': emotions,
        'Count': counts
    })
    
    fig = px.bar(
        df_plot,
        x='Product',
        y='Count',
        color='Emotion',
        barmode='group',
        title='Emotion Distribution per Product (Top 10 Reviews)'
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

# Streamlit UI
st.sidebar.title("Enhanced Semantic Search Engine")
st.sidebar.header("Grupo 7", divider='rainbow')
st.sidebar.markdown("Soria, C., Zurita, M.")
st.sidebar.markdown("Modelo algebraico + Web semántica")

st.header('Amazon Reviews Semantic Search', divider='rainbow')

# Opción para limitar dataset
limit_dataset = st.sidebar.checkbox("Modo rápido (solo primeras 1000 filas)", value=False)
if limit_dataset:
    st.sidebar.warning("⚡ Modo rápido activado para testing")

# Load dataset
uploaded_file = st.file_uploader(
    "Upload the CSV dataset (7817_1.csv)",
    type=["csv"],
    help="Upload the Amazon reviews dataset"
)

if uploaded_file:
    # Leer contenido del archivo y generar hash
    file_content = uploaded_file.read()
    file_hash = get_file_hash(file_content)
    
    # Mostrar información del archivo
    st.info(f"📁 Archivo cargado: {uploaded_file.name} ({len(file_content)/1024/1024:.1f} MB)")
    
    # Procesar datos con cache
    with st.spinner("Procesando datos... (esto puede tomar un momento la primera vez)"):
        processed_data = load_and_process_csv_data(file_content, file_hash)
    
    # Extraer datos procesados
    df = processed_data['df']
    raw_docs = processed_data['raw_docs']
    doc_names = processed_data['doc_names']
    tokenized_docs = processed_data['tokenized_docs']
    entities_list = processed_data['entities_list']
    events_list = processed_data['events_list']
    relations_list = processed_data['relations_list']
    emotions_list = processed_data['emotions_list']
    rdf_graph = processed_data['rdf_graph']
    vocab = processed_data['vocab']
    
    # Mostrar estadísticas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(raw_docs))
    with col2:
        st.metric("Vocabulary Size", len(vocab))
    with col3:
        st.metric("RDF Triplets", len(rdf_graph))
    with col4:
        st.metric("Entities Found", sum(len(e['PRODUCT']) + len(e['BRAND']) for e in entities_list))
    
    # Búsqueda
    st.subheader("🔍 Semantic Search")
    search_query = st.text_input("Enter your search query", placeholder="e.g., kindle problems, great battery life, disappointing purchase")
    
    if search_query:
        # Realizar búsqueda
        with st.spinner("Buscando..."):
            combined_ranked = perform_search(search_query, tokenized_docs, doc_names, raw_docs, emotions_list, vocab)
        
        if combined_ranked:
            # Mostrar resultados
            st.subheader(f"📊 Search Results (Top {len(combined_ranked)})")
            
            # Tabs para diferentes vistas
            tab1, tab2, tab3 = st.tabs(["🔍 Results", "📈 Analytics", "🔗 RDF Insights"])
            
            with tab1:
                for rank, (name, doc, score, emotions) in enumerate(combined_ranked, 1):
                    with st.expander(f"#{rank} {name} - Score: {score:.4f} | Emotions: {emotions}"):
                        highlighted_doc = highlight_terms(doc[:1000] + "..." if len(doc) > 1000 else doc, search_query)
                        st.markdown(highlighted_doc, unsafe_allow_html=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score distribution
                    df_scores = pd.DataFrame({
                        "Review": [name for name, _, _, _ in combined_ranked],
                        "Score": [score for _, _, score, _ in combined_ranked]
                    })
                    st.subheader("Score Distribution")
                    st.bar_chart(df_scores.set_index("Review"))
                
                with col2:
                    # Entity frequency chart
                    entity_fig = create_entity_type_bar_chart(rdf_graph, combined_ranked)
                    st.plotly_chart(entity_fig, use_container_width=True)
                
                # Product emotion chart
                emotion_fig = create_product_emotion_bar_chart(df, emotions_list, combined_ranked)
                st.plotly_chart(emotion_fig, use_container_width=True)
            
            with tab3:
                st.subheader("RDF Triplets Sample")
                triplets_sample = list(rdf_graph)[:20]  # Mostrar solo 20 triplets
                for s, p, o in triplets_sample:
                    st.code(f"({s}, {p}, {o})")
                
                if len(rdf_graph) > 20:
                    st.info(f"Showing 20 of {len(rdf_graph)} total RDF triplets")
        
        else:
            st.warning("No se encontraron resultados relevantes. Intenta con otra consulta.")
    
    # Información adicional en sidebar
    with st.sidebar:
        st.subheader("💡 Tips for better search")
        st.markdown("""
        - Use specific product names (e.g., "kindle paperwhite")
        - Include emotions (e.g., "disappointed", "love", "excellent")
        - Try problem descriptions (e.g., "battery issues", "screen problems")
        - Use purchase-related terms (e.g., "bought", "returned")
        """)
        
        st.subheader("📊 Dataset Info")
        if 'df' in locals():
            st.write(f"Reviews processed: {len(df)}")
            st.write(f"Average rating: {df['reviews.rating'].mean():.2f}")
            st.write(f"Products: {df['name'].nunique()}")

else:
    st.info("📤 Upload the CSV dataset to begin the search.")
    st.markdown("""
    ### Features:
    - **Smart Caching**: Data is processed once and cached for faster subsequent loads
    - **Progress Tracking**: Real-time progress during processing
    - **Batch Processing**: Optimized NLP processing in batches
    - **Memory Efficient**: Reduced memory usage with optimized data structures
    - **Fast Search**: BM25 + Cosine similarity with caching
    - **Interactive UI**: Tabbed interface with analytics and insights
    """)