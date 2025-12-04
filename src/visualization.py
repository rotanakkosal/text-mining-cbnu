"""
Visualization module for the text mining project.
Generates plots for dataset exploration and results analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def plot_rating_distribution(df, output_dir='outputs'):
    """Plot the distribution of star ratings in the dataset."""
    rating_counts = df['Rating'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff6b6b', '#ffa500', '#808080', '#87ceeb', '#90ee90']
    bars = plt.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Star Rating', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Reviews', fontsize=12, fontweight='bold')
    plt.title('Distribution of Star Ratings in Dataset', fontsize=14, fontweight='bold', pad=20)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(rating_counts.items()):
        plt.text(idx, val + 2000, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/1_rating_distribution.png")

def plot_class_distribution(df, output_dir='outputs'):
    """Plot the class distribution after filtering 3-star reviews."""
    df_filtered = df[df['Rating'] != 3].copy()
    df_filtered['label'] = np.where(df_filtered['Rating'] <= 2, 'Defect/Complaint', 'Praise/Noise')
    class_counts = df_filtered['label'].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = ['#ff6b6b', '#51cf66']
    wedges, texts, autotexts = plt.pie(
        class_counts.values, 
        labels=class_counts.index, 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        explode=(0.05, 0.05),
        shadow=True
    )
    
    # Make percentage text larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    plt.title('Class Distribution After Filtering 3-Star Reviews', fontsize=14, fontweight='bold', pad=20)
    
    # Add count information
    total = class_counts.sum()
    plt.text(0, -1.3, f'Total Reviews: {total:,}', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/2_class_distribution.png")

def plot_scorecard(scorecard_path, output_dir='outputs'):
    """Plot the prioritized scorecard results."""
    scorecard = pd.read_csv(scorecard_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Issue Count
    colors1 = plt.cm.Reds(np.linspace(0.5, 0.9, len(scorecard)))
    ax1.barh(scorecard['Category'], scorecard['Issue_Count'], color=colors1, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Number of Issues', fontsize=12, fontweight='bold')
    ax1.set_title('Complaint Categories by Frequency', fontsize=14, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(scorecard['Issue_Count']):
        ax1.text(v + 1000, i, f'{v:,}', va='center', fontsize=10, fontweight='bold')
    
    # Average Anger Score
    colors2 = plt.cm.Oranges(np.linspace(0.5, 0.9, len(scorecard)))
    ax2.barh(scorecard['Category'], scorecard['Avg_Anger_Score'], color=colors2, edgecolor='black', linewidth=1)
    ax2.set_xlabel('Average Anger Score', fontsize=12, fontweight='bold')
    ax2.set_title('Complaint Categories by Customer Frustration', fontsize=14, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(scorecard['Avg_Anger_Score']):
        ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_scorecard_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/3_scorecard_visualization.png")

def plot_model_comparison(output_dir='outputs'):
    """Plot model performance comparison."""
    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost'],
        'Train Accuracy': [0.9281, 0.9408],
        'Test Accuracy': [0.9260, 0.9337],
        'Train F1': [0.8545, 0.8810],
        'Test F1': [0.8499, 0.8666]
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    metrics = ['Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1']
    colors = ['#4ecdc4', '#45b7d1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(results['Model'], results[metric], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0.8, 0.95])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(results[metric]):
            ax.text(i, v + 0.003, f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Highlight best model
        best_idx = results[metric].idxmax()
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/4_model_comparison.png")

def plot_overfitting_analysis(output_dir='outputs'):
    """Plot train vs test performance to check for overfitting."""
    metrics = {
        'Metric': ['Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1'],
        'Logistic Regression': [0.9281, 0.9260, 0.8545, 0.8499],
        'XGBoost': [0.9408, 0.9337, 0.8810, 0.8666]
    }
    
    x = np.arange(len(metrics['Metric']))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, metrics['Logistic Regression'], width, 
                   label='Logistic Regression', color='#4ecdc4', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, metrics['XGBoost'], width, 
                   label='XGBoost', color='#45b7d1', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Test Performance (Overfitting Analysis)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['Metric'], fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0.8, 0.95])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_overfitting_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/5_overfitting_analysis.png")

def plot_vector_space(X, y, output_dir='outputs', sample_size=5000):
    """
    Visualize data in 2D vector space using PCA.
    Shows data points and decision boundary.
    
    Parameters:
    - X: Feature matrix (sparse or dense)
    - y: Labels (0 or 1)
    - output_dir: Output directory for saving plot
    - sample_size: Number of samples to plot (for performance)
    """
    print("   - Creating vector space visualization (PCA 2D projection)...")
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # Sample data if too large (for performance)
    if X_dense.shape[0] > sample_size:
        indices = np.random.choice(X_dense.shape[0], sample_size, replace=False)
        X_sample = X_dense[indices]
        y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        print(f"      (Sampling {sample_size} points from {X_dense.shape[0]} for visualization)")
    else:
        X_sample = X_dense
        y_sample = y
    
    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    
    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_
    total_variance = variance_explained.sum()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Data points colored by class
    defect_mask = (y_sample == 1)
    praise_mask = (y_sample == 0)
    
    ax1.scatter(X_2d[praise_mask, 0], X_2d[praise_mask, 1], 
               c='#51cf66', label='Praise/Noise (Class 0)', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax1.scatter(X_2d[defect_mask, 0], X_2d[defect_mask, 1], 
               c='#ff6b6b', label='Defect/Complaint (Class 1)', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel(f'First Principal Component (PC1)\nExplains {variance_explained[0]*100:.1f}% variance', 
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'Second Principal Component (PC2)\nExplains {variance_explained[1]*100:.1f}% variance', 
                   fontsize=11, fontweight='bold')
    ax1.set_title(f'Data in 2D Vector Space (PCA Projection)\nTotal Variance Explained: {total_variance*100:.1f}%', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Data points with decision boundary
    # Train XGBoost on 2D data to show decision boundary (matching the actual best model)
    xgb_2d = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=100)
    xgb_2d.fit(X_2d, y_sample)
    
    # Create a mesh for decision boundary
    h = 0.02  # Step size in mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = xgb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax2.contourf(xx, yy, Z, alpha=0.3, colors=['#51cf66', '#ff6b6b'], levels=[0, 0.5, 1])
    ax2.contour(xx, yy, Z, colors='black', linewidths=2, linestyles='--', alpha=0.5)
    
    # Plot data points
    ax2.scatter(X_2d[praise_mask, 0], X_2d[praise_mask, 1], 
               c='#51cf66', label='Praise/Noise (Class 0)', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax2.scatter(X_2d[defect_mask, 0], X_2d[defect_mask, 1], 
               c='#ff6b6b', label='Defect/Complaint (Class 1)', alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel(f'First Principal Component (PC1)\nExplains {variance_explained[0]*100:.1f}% variance', 
                   fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'Second Principal Component (PC2)\nExplains {variance_explained[1]*100:.1f}% variance', 
                   fontsize=11, fontweight='bold')
    ax2.set_title('Data with Decision Boundary (XGBoost on 2D PCA Projection)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_vector_space_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_dir}/6_vector_space_2d.png")
    print(f"      Note: PCA explains {total_variance*100:.1f}% of variance in 2D projection")

def generate_all_visualizations(df, scorecard_path='outputs/prioritized_scorecard.csv', output_dir='outputs', 
                                X_all_vectors=None, y_labels=None):
    """Generate all visualizations for the project."""
    print("\n📊 Generating Visualizations...")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    plot_rating_distribution(df, output_dir)
    plot_class_distribution(df, output_dir)
    
    if os.path.exists(scorecard_path):
        plot_scorecard(scorecard_path, output_dir)
    else:
        print(f"   ⚠️  Scorecard not found at {scorecard_path}, skipping scorecard plot")
    
    plot_model_comparison(output_dir)
    plot_overfitting_analysis(output_dir)
    
    # Vector space visualization (if data provided)
    if X_all_vectors is not None and y_labels is not None:
        plot_vector_space(X_all_vectors, y_labels, output_dir)
    else:
        print("   ⚠️  Skipping vector space plot (data not provided)")
    
    print("\n✅ All visualizations generated successfully!")
    print(f"   📁 Saved to: {output_dir}/")

