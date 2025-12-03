import pandas as pd
import numpy as np
import os

def load_amazon_data(filepath):
    """
    Loads the full Amazon dataset, filters neutral reviews, and creates binary labels.
    """
    print(f"🔄 [Ingestion] Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found at {filepath}. Please check the path.")

    # Load the full dataset
    df = pd.read_csv(filepath)

    # Filter: Drop neutral 3-star reviews (Ambiguous data)
    df = df[df['Rating'] != 3].copy()

    # Labeling: 
    # 1 & 2 Stars = Class 1 (Defect/Complaint)
    # 4 & 5 Stars = Class 0 (Noise/Praise)
    df['label'] = np.where(df['Rating'] <= 2, 1, 0)

    print(f"✅ [Ingestion] Loaded {len(df)} rows.")
    print(f"   - Defects (Class 1): {sum(df['label'] == 1)}")
    print(f"   - Noise   (Class 0): {sum(df['label'] == 0)}")
    
    return df