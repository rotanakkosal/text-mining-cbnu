import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def compare_and_train(X_train, X_test, y_train, y_test):
    """
    Compares different models and returns the best one.
    """
    print("\n📊 [Modeling] Comparing Models...")
    
    # --- Model Definitions ---
    # Removed MultinomialNB as it cannot handle negative VADER scores.
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = []
    
    for name, model in models.items():
        print(f"   - Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on Test Set
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        
        # Evaluate on Training Set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        results.append({
            'model': name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1
        })
    
    # --- Display Results ---
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Find the best model based on F1-score on the test set
    best_model_name = max(results, key=lambda x: x['test_f1'])['model']
    best_f1 = max(results, key=lambda x: x['test_f1'])['test_f1']
    
    print(f"\n🏆 Best Model Selected: {best_model_name} with Test F1-Score: {best_f1:.4f}")
    
    # Return the actual trained model object, not just its name
    best_model_object = models[best_model_name]
    return best_model_object

def generate_scorecard(df, model, vectorizer):
    """
    Generates a scorecard based on the custom-trained model.
    """
    print("\n🚀 [Scorecard] Generating Prioritized Fixes...")
    
    # 1. Get text features
    X_tfidf = vectorizer.transform(df['clean_text'])
    
    # 2. Get VADER score features
    vader_scores = df['compound_score'].values.reshape(-1, 1)
    
    # 3. Combine them into the final feature set
    X_all_combined = hstack([X_tfidf, vader_scores]).tocsr()
    
    # Predict on the entire dataset to categorize all reviews
    all_predictions = model.predict(X_all_combined)
    df['predicted_label'] = all_predictions
    
    # Filter for only the defects/complaints
    defects = df[df['predicted_label'] == 1].copy()
    
    # 3. Sentiment Analysis (How angry are they?)
    analyzer = SentimentIntensityAnalyzer()
    defects['neg_sentiment'] = defects['clean_text'].apply(
        lambda x: analyzer.polarity_scores(x)['neg']
    )
    
    # 4. Keyword Categorization (Simple Topic Grouping)
    def categorize(text):
        if 'battery' in text or 'charge' in text or 'power' in text: return 'Battery/Power'
        if 'screen' in text or 'display' in text or 'pixel' in text: return 'Screen/Display'
        if 'sound' in text or 'speaker' in text or 'audio' in text: return 'Audio/Mic'
        if 'sim' in text or 'card' in text or 'signal' in text: return 'Connectivity'
        if 'slow' in text or 'freeze' in text or 'crash' in text: return 'Performance/UI'
        return 'General/Other'

    defects['Category'] = defects['clean_text'].apply(categorize)
    
    # 5. Aggregate
    scorecard = defects.groupby('Category').agg(
        Issue_Count=('Category', 'count'),
        Avg_Anger_Score=('neg_sentiment', 'mean')
    ).sort_values(by='Issue_Count', ascending=False)
    
    return scorecard