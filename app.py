import streamlit as st
import os
import re
import numpy as np
import pandas as pd
import unicodedata
from nltk.corpus import stopwords
from nltk import download
import spacy
from rank_bm25 import BM25Okapi
from rdflib import Graph, RDF, RDFS, Literal, URIRef
from functools import reduce
from collections import defaultdict
import json
import plotly.express as px

# Download NLTK stopwords if not present
download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))
english_stopwords = set(stopwords.words('english'))
stopwords_set = spanish_stopwords | english_stopwords

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install the spaCy model by running: python -m spacy download en_core_web_sm")
    st.stop()

# Step 1: Dataset loading
def load_csv_data(file_path):
    """Load and preprocess CSV data"""
    df = pd.read_csv(file_path)
    df['text_content'] = df['reviews.text'].fillna('') + ' ' + df['reviews.title'].fillna('')
    return df

# Step 2: Cleaning + Regex
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

# Step 3: Named Entity Recognition (NER)
def extract_entities(text):
    """Extract entities using spaCy"""
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

# Step 4: Event Extraction
def extract_events(text):
    """Extract events based on key verbs"""
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

# Step 5: Relation Extraction
def extract_relations(text, entities):
    """Extract relations between entities"""
    relations = []
    doc = nlp(text)
    
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

# Step 6: Emotion Extraction
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

# Step 7: Knowledge Representation (RDF Triplets)
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

# Step 8: Semantic Web and Search with BM25
def bm25_search(query, tokenized_docs, doc_names):
    """Perform BM25 search on tokenized documents"""
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = clean_words(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(
        [(name, doc, score) for name, doc, score in zip(doc_names, tokenized_docs, scores) if score > 0],
        key=lambda x: x[2], reverse=True
    )
    return ranked

# Step 9: Entity Type Frequency Bar Chart
def create_entity_type_bar_chart(rdf_graph, ranked_reviews):
    """Create a bar chart showing the frequency of entity types in the RDF graph"""
    namespace = "http://example.org/amazon_reviews#"
    entity_counts = defaultdict(int)
    
    # Count entity types
    for s, p, o in rdf_graph:
        if str(p).endswith('type'):
            entity_type = str(o).split('#')[-1]
            entity_counts[entity_type] += 1
    
    # Highlight top-ranked reviews
    top_reviews = {name for name, _, _, _ in ranked_reviews[:5]}  # Updated to unpack 4 elements
    highlighted_counts = defaultdict(int)
    for idx, row in df.iterrows():
        if f"Review_{idx+1}" in top_reviews:
            for entity_type, entities in entities_list[idx].items():
                highlighted_counts[entity_type] += len(entities)
    
    # Prepare data for bar chart
    entity_types = list(entity_counts.keys())
    counts = [entity_counts[et] for et in entity_types]
    highlight_counts = [highlighted_counts.get(et, 0) for et in entity_types]
    
    df_plot = pd.DataFrame({
        'Entity Type': entity_types,
        'Total Count': counts,
        'Highlighted Count (Top Reviews)': highlight_counts
    })
    
    fig = px.bar(
        df_plot,
        x='Entity Type',
        y=['Total Count', 'Highlighted Count (Top Reviews)'],
        barmode='group',
        title='Entity Type Frequency in Reviews',
        labels={'value': 'Count', 'variable': 'Count Type'}
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

# Step 10: Product-Emotion Bar Chart
def create_product_emotion_bar_chart(df, emotions_list, ranked_reviews):
    """Create a bar chart showing emotion counts per product for top-ranked reviews"""
    top_reviews = {name for name, _, _, _ in ranked_reviews[:10]}  # Updated to unpack 4 elements
    emotion_counts = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        if f"Review_{idx+1}" in top_reviews:
            product = row['name']
            for emotion in emotions_list[idx]:
                emotion_counts[product][emotion] += 1
    
    # Prepare data for bar chart
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
        title='Emotion Distribution per Product (Top 10 Reviews)',
        labels={'Count': 'Emotion Count'}
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

# Helper functions
def tokenize(text):
    """Divide text into words using regex"""
    return re.findall(r'\b\w+\b', text)

def clean_words(text):
    """Tokenize, normalize, and remove stopwords"""
    tokens = tokenize(normalize_text(text))
    return [word for word in tokens if word not in stopwords_set and word.isalpha()]

def normalize_text(text):
    """Convert text to lowercase and remove accents"""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vectorize(words, vocab):
    """Convert list of words to binary vector based on vocabulary"""
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)

def highlight_terms(text, query):
    """Highlight query terms in text"""
    terms = set(clean_words(query))
    for term in sorted(terms, key=len, reverse=True):
        if term:
            pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
            text = pattern.sub(r'<mark>\1</mark>', text)
    return text

# Streamlit UI
st.sidebar.title("Enhanced Semantic Search Engine")
st.sidebar.header("Grupo 7", divider='rainbow')
st.sidebar.markdown("Soria, C., Zurita, M.")
st.sidebar.markdown("Modelo algebraico + Web semántica")

st.header('Amazon Reviews Semantic Search', divider='rainbow')

# Load dataset
uploaded_file = st.file_uploader(
    "Upload the CSV dataset (7817_1.csv)",
    type=["csv"],
    help="Upload the Amazon reviews dataset"
)

if uploaded_file:
    # Step 1: Load dataset
    st.subheader("Step 1: Dataset Loading")
    st.write("Loading the Amazon reviews dataset from the uploaded CSV file.")
    df = load_csv_data(uploaded_file)
    raw_docs = df['text_content'].tolist()
    doc_names = [f"Review_{i+1}" for i in range(len(raw_docs))]
    
    with st.expander("View console output"):
        step_one_output = ["1. Loading Amazon reviews dataset:"]
        step_one_output.append(f"Total reviews: {len(raw_docs)}")
        step_one_output.append(f"Sample review (first 50 chars): {raw_docs[0][:50]}...")
        st.code("\n".join(step_one_output), language='plaintext')

    # Step 2: Cleaning and Regex
    st.subheader("Step 2: Cleaning and Regex Extraction")
    st.write("Cleaning text, extracting prices and product models using regex.")
    cleaned_docs = []
    prices_list = []
    models_list = []
    for doc in raw_docs:
        cleaned, prices, models = clean_text(doc)
        cleaned_docs.append(cleaned)
        prices_list.append(prices)
        models_list.append(models)
    
    tokenized_docs = [clean_words(doc) for doc in cleaned_docs]
    
    with st.expander("View console output"):
        step_two_output = ["2. Cleaning and extracting prices/models:"]
        for i, (doc, prices, models) in enumerate(zip(cleaned_docs, prices_list, models_list), 1):
            step_two_output.append(f"Review {i}: {len(doc.split())} words, Prices: {prices[:3]}, Models: {models[:3]}")
        st.code("\n".join(step_two_output), language='plaintext')

    # Step 3: NER
    st.subheader("Step 3: Named Entity Recognition")
    st.write("Extracting entities (PRODUCT, BRAND, LOCATION, PERSON) using spaCy.")
    entities_list = [extract_entities(doc) for doc in raw_docs]
    
    with st.expander("View console output"):
        step_three_output = ["3. Extracted entities:"]
        for i, entities in enumerate(entities_list, 1):
            step_three_output.append(f"Review {i}: {entities}")
        st.code("\n".join(step_three_output), language='plaintext')

    # Step 4: Event Extraction
    st.subheader("Step 4: Event Extraction")
    st.write("Extracting events based on key verbs (e.g., bought, failed).")
    events_list = [extract_events(doc) for doc in raw_docs]
    
    with st.expander("View console output"):
        step_four_output = ["4. Extracted events:"]
        for i, events in enumerate(events_list, 1):
            step_four_output.append(f"Review {i}: {events}")
        st.code("\n".join(step_four_output), language='plaintext')

    # Step 5: Relation Extraction
    st.subheader("Step 5: Relation Extraction")
    st.write("Extracting relations (e.g., PERSON recommended PRODUCT).")
    relations_list = [extract_relations(doc, entities) for doc, entities in zip(raw_docs, entities_list)]
    
    with st.expander("View console output"):
        step_five_output = ["5. Extracted relations:"]
        for i, relations in enumerate(relations_list, 1):
            step_five_output.append(f"Review {i}: {relations}")
        st.code("\n".join(step_five_output), language='plaintext')

    # Step 6: Emotion Extraction
    st.subheader("Step 6: Emotion Extraction")
    st.write("Extracting emotions based on sentiment and keywords.")
    emotions_list = [extract_emotions(doc, row[1]['reviews.rating']) for row, doc in zip(df.iterrows(), raw_docs)]
    
    with st.expander("View console output"):
        step_six_output = ["6. Extracted emotions:"]
        for i, emotions in enumerate(emotions_list, 1):
            step_six_output.append(f"Review {i}: {emotions}")
        st.code("\n".join(step_six_output), language='plaintext')

    # Step 7: Knowledge Representation (RDF)
    st.subheader("Step 7: Knowledge Representation")
    st.write("Creating RDF triplets for entities, events, relations, and emotions.")
    rdf_graph = create_rdf_triplets(df, entities_list, events_list, relations_list, emotions_list)
    
    with st.expander("View RDF triplets"):
        step_seven_output = ["7. RDF Triplets:"]
        for s, p, o in rdf_graph:
            step_seven_output.append(f"({s}, {p}, {o})")
        st.code("\n".join(step_seven_output[:10]) + "\n...", language='plaintext')

    # Step 8: Semantic Search with BM25
    st.subheader("Step 8: Semantic Search")
    st.write("Performing search using BM25 ranking. Enter a query to search reviews.")
    search_query = st.text_input("Enter your search query")
    
    if search_query:
        # BM25 Search
        ranked = bm25_search(search_query, tokenized_docs, doc_names)
        
        # Cosine Similarity
        vocab = sorted(set(reduce(lambda x, y: x + y, tokenized_docs, clean_words(search_query))))
        doc_vectors = [vectorize(doc, vocab) for doc in tokenized_docs]
        query_vector = vectorize(clean_words(search_query), vocab)
        cosine_scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors]
        
        # Combine BM25 and Cosine scores
        combined_ranked = []
        for i, (name, doc, bm25_score) in enumerate(ranked):
            cosine_score = cosine_scores[doc_names.index(name)]
            combined_score = 0.7 * bm25_score + 0.3 * cosine_score
            review_idx = doc_names.index(name)
            combined_ranked.append((name, raw_docs[review_idx], combined_score, emotions_list[review_idx]))
        
        combined_ranked = sorted(combined_ranked, key=lambda x: x[2], reverse=True)[:10]  # Limit to top 10
        
        with st.expander("View console output"):
            step_eight_output = ["8. Search results (BM25 + Cosine, Top 10):"]
            for rank, (name, doc, score, emotions) in enumerate(combined_ranked, 1):
                step_eight_output.append(f"{rank}. {name} - Combined Score: {score:.4f}, Emotions: {emotions}")
            st.code("\n".join(step_eight_output), language='plaintext')

        # Step 9: Entity Type Frequency Bar Chart
        st.subheader("Step 9: Entity Type Frequency")
        st.write("Bar chart showing frequency of entity types (e.g., Product, Emotion) in reviews, with top-ranked reviews highlighted.")
        entity_fig = create_entity_type_bar_chart(rdf_graph, combined_ranked)
        st.plotly_chart(entity_fig, use_container_width=True)

        # Step 10: Product-Emotion Bar Chart
        st.subheader("Step 10: Product-Emotion Distribution")
        st.write("Bar chart showing emotion counts per product for top 10 ranked reviews.")
        emotion_fig = create_product_emotion_bar_chart(df, emotions_list, combined_ranked)
        st.plotly_chart(emotion_fig, use_container_width=True)

        # Search Results (Top 10)
        st.subheader("Search Results (Top 10)")
        if combined_ranked:
            for rank, (name, doc, score, emotions) in enumerate(combined_ranked, 1):
                st.write(f"{rank}. {name} - Combined Score: {score:.4f}, Emotions: {emotions}")
                highlighted_doc = highlight_terms(doc, search_query)
                with st.expander(f"View review content"):
                    st.markdown(
                        f'<div style="max-height: 200px; overflow-y: auto; background: #f9f9f9; padding: 8px; border-radius: 4px; white-space: pre-wrap;">{highlighted_doc}</div>',
                        unsafe_allow_html=True
                    )
            
            # Score Distribution
            df_sim = pd.DataFrame({
                "Review": [name for name, _, _, _ in combined_ranked],
                "Combined Score": [score for _, _, score, _ in combined_ranked]
            })
            df_sim["Review"] = pd.Categorical(df_sim["Review"], categories=df_sim["Review"], ordered=True)
            df_sim = df_sim.set_index("Review")
            
            st.subheader("Score Distribution")
            st.bar_chart(df_sim)
        else:
            st.info("No relevant reviews found. Try a different query.")
else:
    st.info("Upload the CSV dataset to begin the search.")