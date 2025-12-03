import os
import sys
import joblib
# Import modules from the src folder
from data_loader import load_amazon_data
from preprocessing import process_text_data
from modeling import compare_and_train, generate_scorecard

def main():
    # --- CONFIGURATION ---
    DATA_PATH = "data/Amazon_Unlocked_Mobile.csv"
    OUTPUT_PATH = "outputs/prioritized_scorecard.csv"
    MODEL_DIR = "model"
    
    # Ensure output folders exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # 1. Load Data
        df = load_amazon_data(DATA_PATH)
        
        # 2. Preprocessing
        X_all_vectors, X_train, X_test, y_train, y_test, vectorizer = process_text_data(df)
        
        # 3. Modeling & Comparison
        best_model = compare_and_train(X_train, X_test, y_train, y_test)
        
        # 4. Save the model and vectorizer
        print(f"\n💾 Saving model and vectorizer to {MODEL_DIR}...")
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
        print("✅ Model and vectorizer saved successfully.")

        # 5. Generate Scorecard (Business Intelligence)
        final_scorecard = generate_scorecard(df, best_model, vectorizer)
        
        # 6. Save Results
        print("\n🏆 FINAL PRIORITIZED SCORECARD 🏆")
        print(final_scorecard)
        final_scorecard.to_csv(OUTPUT_PATH)
        print(f"\n✅ Result saved to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()