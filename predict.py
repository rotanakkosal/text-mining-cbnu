import joblib
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack

# --- CONFIGURATION ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

def clean_text(text):
    """Basic text normalization (must be identical to training)."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def predict_sentiment(review_text):
    """
    Loads the trained model and vectorizer to predict the sentiment of a new review.
    Returns the binary prediction and the VADER sentiment score.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return "❌ Error: Model or vectorizer not found.", {}

    # Load the model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Clean the input text exactly as it was done in training
    cleaned_review = clean_text(review_text)
    
    # Get VADER score (using the cleaned text is slightly better)
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = analyzer.polarity_scores(cleaned_review)
    compound_score = vader_scores['compound']

    # 1. Transform the CLEANED review text
    vectorized_text = vectorizer.transform([cleaned_review])
    
    # 2. Add the VADER score as the second feature
    vader_feature = [[compound_score]]
    
    # 3. Combine them into a single feature set
    combined_features = hstack([vectorized_text, vader_feature]).tocsr()
    
    # Make a prediction using the combined features
    prediction = model.predict(combined_features)
    
    # Interpret the prediction
    if prediction[0] == 1:
        binary_prediction = "😠 Negative (Complaint)"
    else:
        binary_prediction = "😊 Positive (Praise)"
        
    return binary_prediction, compound_score

if __name__ == "__main__":
    # --- Example Usage ---
    reviews_to_test = [
        "The battery life on this phone is not good, it barely lasts a few hours.",
        "I love the camera! The pictures are so clear and vibrant.",
        "The phone is better than the previous one.",
        "Does not support newest or newer iOS. I did not know that when p[purchasing as a replacement for my wife. We are not apple users (this was just the cheapest). She hates it, thus I hate it..."
    ]
    
    for i, review in enumerate(reviews_to_test, 1):
        prediction, compound_score = predict_sentiment(review)
        print(f"Review #{i}: '{review}'")
        print(f"   - Model Prediction: {prediction}")
        print(f"   - VADER Score (Sentiment): {compound_score}\n")