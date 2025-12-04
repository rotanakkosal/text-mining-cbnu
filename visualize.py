"""
Standalone visualization script.
Run this independently to generate all visualizations: python visualize.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_amazon_data
from src.visualization import generate_all_visualizations

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION GENERATOR")
    print("=" * 60)
    
    # Load data
    df = load_amazon_data("data/Amazon_Unlocked_Mobile.csv")
    
    # Generate all visualizations
    scorecard_path = "outputs/prioritized_scorecard.csv"
    generate_all_visualizations(df, scorecard_path, 'outputs')
    
    print("\n" + "=" * 60)
    print("Visualization generation complete!")
    print("=" * 60)

