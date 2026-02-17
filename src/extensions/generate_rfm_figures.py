
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import roc_curve, auc
from src.rfm import calculate_rfm, create_proxy_target
from src.config import config
from src.data_processing import load_data, preprocessing_pipeline

# Set plot style
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Ensure output directory exists
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

def generate_rfm_visuals():
    """Generate visualizations for RFM analysis and Proxy Target labeling."""
    print("--- Generating RFM & Proxy Label Visuals ---")
    
    # 1. Load data and generate RFM
    raw_data = load_data(str(config.paths.raw_data))
    rfm_df = calculate_rfm(raw_data)
    labeled_df = create_proxy_target(rfm_df)
    
    # 2. RFM 3D Scatter Plot (using 2D projections for simplicity in PNG)
    plt.figure(figsize=(15, 6))
    
    # Recency vs Frequency
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=labeled_df, x='Recency', y='Frequency', hue='Risk_Label', palette='viridis', alpha=0.6)
    plt.title('Recency vs Frequency by Risk Label')
    
    # Frequency vs Monetary
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=labeled_df, x='Frequency', y='Monetary', hue='Risk_Label', palette='viridis', alpha=0.6)
    plt.title('Frequency vs Monetary by Risk Label')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rfm_clusters.png')
    plt.close()
    print(f"✓ Saved rfm_clusters.png")

    # 3. Distribution of Risk Labels
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Risk_Label', data=labeled_df, palette=['#55a868', '#c44e52'])
    plt.title('Distribution of Customer Risk Labels (Proxy Target)')
    plt.xticks([0, 1], ['Low Risk (0)', 'High Risk (1)'])
    plt.savefig(f'{output_dir}/risk_label_distribution.png')
    plt.close()
    print(f"✓ Saved risk_label_distribution.png")

def generate_model_performance_visuals():
    """Generate visualizations for Model Performance (ROC-AUC)."""
    print("\n--- Generating Model Performance Visuals ---")
    
    # Path to models
    models_dir = 'models'
    lr_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    raw_path = str(config.paths.raw_data)
    
    if not (os.path.exists(lr_path) and os.path.exists(rf_path)):
        print("Models not found. Skip generating performance visuals. Run training first.")
        return

    # Load artifacts
    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Prepare data for testing
    df = load_data(raw_path)
    rfm_df = calculate_rfm(df)
    labeled_df = create_proxy_target(rfm_df)
    
    X = labeled_df[['Recency', 'Frequency', 'Monetary']]
    y = labeled_df['Risk_Label']
    
    X_processed = preprocessor.transform(X)
    
    # Generate ROC Curves
    plt.figure(figsize=(10, 8))
    
    # Logistic Regression ROC
    lr_probs = lr.predict_proba(X_processed)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_probs)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.4f})', color='blue', lw=2)
    
    # Random Forest ROC
    rf_probs = rf.predict_proba(X_processed)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y, rf_probs)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})', color='green', lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/model_performance_roc.png')
    plt.close()
    print(f"✓ Saved model_performance_roc.png")
    
    # Feature Importance (Random Forest)
    plt.figure(figsize=(10, 6))
    importances = rf.feature_importances_
    features = ['Recency', 'Frequency', 'Monetary']
    indices = np.argsort(importances)
    
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance (Random Forest Model)')
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    print(f"✓ Saved feature_importance.png")

if __name__ == "__main__":
    generate_rfm_visuals()
    generate_model_performance_visuals()
    print("\n✅ All RFM and ML figures generated successfully!")
