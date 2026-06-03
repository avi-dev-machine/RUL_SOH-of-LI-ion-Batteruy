"""
Additional Graphical Metrics & Confusion Matrix
===============================================
Generates graphical performance metrics and a binned confusion matrix
for the regression outputs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
MODELS_DIR = os.path.join(THIS_DIR, 'saved_models')

# Import modules from hybrid_model
from data_loader import load_dataset, add_derived_features, split_data
from feature_fusion import fuse_features, FeatureScaler
from physics_informed_ann import PhysicsInformedBatteryANN
from evaluation.metrics import compute_full_metrics

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def generate_performance_bar_charts(metrics_df: pd.DataFrame):
    """Plots MAE, RMSE, and R2 as bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Graphical Performance Metrics (Test Set)', fontsize=16, fontweight='bold', y=1.05)
    
    colors = ['#3498DB', '#E74C3C'] # Blue for SOH, Red for RUL
    targets = metrics_df['Target'].values
    
    # MAE
    axes[0].bar(targets, metrics_df['MAE'], color=colors, alpha=0.8)
    axes[0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    for i, v in enumerate(metrics_df['MAE']):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')
        
    # RMSE
    axes[1].bar(targets, metrics_df['RMSE'], color=colors, alpha=0.8)
    axes[1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    for i, v in enumerate(metrics_df['RMSE']):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')
        
    # R2
    axes[2].bar(targets, metrics_df['R2'], color=colors, alpha=0.8)
    axes[2].set_title('R² Score', fontweight='bold')
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(metrics_df['R2']):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'graphical_metrics_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved bar charts -> {path}")
    plt.close()


def generate_soh_confusion_matrix(y_true_soh: np.ndarray, y_pred_soh: np.ndarray):
    """
    Since this is regression, we bin SOH into categories to create a confusion matrix:
    - Healthy: > 90%
    - Warning: 80% to 90%
    - EOL (End of Life): < 80%
    """
    bins = [0, 80, 90, 150]
    labels = ['EOL (<80%)', 'Warning (80-90%)', 'Healthy (>90%)']
    
    true_binned = pd.cut(y_true_soh, bins=bins, labels=labels)
    pred_binned = pd.cut(y_pred_soh, bins=bins, labels=labels)
    
    cm = confusion_matrix(true_binned, pred_binned, labels=labels)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Custom plotting with seaborn for better colors
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_xlabel('Predicted State', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual State', fontsize=12, fontweight='bold')
    ax.set_title('SOH State Classification Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'soh_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix -> {path}")
    plt.close()


def main():
    print("Loading data & model...")
    df = load_dataset()
    df = add_derived_features(df)
    
    # We must load the Physics Engine to get the physics features for fusion
    from physics_engine import PhysicsEngine
    engine = PhysicsEngine().fit(df)
    df = engine.transform(df)
    
    # Fusion
    X_fused, feature_cols = fuse_features(df, use_physics=True)
    
    # Split
    data = split_data(df, feature_cols, test_size=0.20, val_size=0.10, random_state=42)
    X_test = data['X_test']
    y_test = data['y_test']
    y_phys_test = data['y_phys_test']
    
    # Scaler
    scaler = FeatureScaler().load(MODELS_DIR)
    X_test_s = scaler.transform_X(X_test)
    
    # Model
    ann = PhysicsInformedBatteryANN(
        input_dim=X_test_s.shape[1],
        trunk_units=[256, 128, 64],
        head_units=[32, 16]
    ).load(MODELS_DIR, tag='best')
    
    # Predict
    print("Generating predictions...")
    y_pred_s = ann.predict(X_test_s)
    y_pred = scaler.inverse_y(y_pred_s)
    
    # Metrics
    metrics = compute_full_metrics(
        y_test, y_pred, y_phys_test[:, 0], 
        data['meta_test']['cycle'].values, 
        data['meta_test']['battery_id'].values
    )
    
    metrics_df = metrics['summary']
    
    print("Plotting graphical metrics...")
    generate_performance_bar_charts(metrics_df)
    
    print("Plotting SOH confusion matrix...")
    generate_soh_confusion_matrix(y_test[:, 0], y_pred[:, 0])

if __name__ == '__main__':
    main()
