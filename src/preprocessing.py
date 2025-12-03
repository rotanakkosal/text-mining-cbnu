import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack

def clean_text(text):
    """Basic text normalization."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def process_text_data(df):
    """
    Cleans text and converts it to vectors (CountVectorizer).
    Returns X_train, X_test, y_train, y_test, and the vectorizer.
    """
    print("⚙️ [Preprocessing] Cleaning text and vectorizing...")
    
    # Apply the cleaning function to the 'Reviews' column
    df['clean_text'] = df['Reviews'].fillna('').apply(clean_text)
    
    # Add VADER sentiment scores
    analyzer = SentimentIntensityAnalyzer()
    df['compound_score'] = df['clean_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

    # Text Vectorization (TF-IDF)
    # Allow the model to consider both single words and pairs of words.
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    
    # Add VADER score as a new feature
    print("✨ [Feature Engineering] Adding VADER score as a new feature...")
    vader_scores = df['compound_score'].values.reshape(-1, 1)
    X_all_vectors = hstack([X_tfidf, vader_scores]).tocsr()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all_vectors, df['label'], test_size=0.2, random_state=42
    )
    
    return X_all_vectors, X_train, X_test, y_train, y_test, vectorizer