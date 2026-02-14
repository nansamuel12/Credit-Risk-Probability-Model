import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

from src.data_processing import load_data, aggregate_features, preprocess_pipeline
from src.rfm import calculate_rfm, create_proxy_target
from src.config import config

def run_training_pipeline():
    """Main training pipeline following interim requirements."""
    
    # 1. Load Data
    print("--- Loading Raw Data ---")
    raw_path = str(config.paths.raw_data)
    df = load_data(raw_path)
    
    # 2. Engineering RFM and Proxy Target
    print("--- Engineering RFM Features & Proxy Target ---")
    df_rfm = calculate_rfm(df)
    df_labeled = create_proxy_target(df_rfm, random_state=42)
    
    # 3. Join with extra aggregated features if needed
    # (For simplicity and requirement adherence, we'll use the RFM features as our model input)
    X = df_labeled[['Recency', 'Frequency', 'Monetary']]
    y = df_labeled['Risk_Label']
    
    # 4. Train/Test Split
    print("--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Preprocessing (Scaling numeric cols)
    print("--- Preprocessing ---")
    preprocessor = preprocess_pipeline(num_cols=['Recency', 'Frequency', 'Monetary'])
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # 6. Train Models
    print("--- Training Models ---")
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    
    # 7. Print Metrics (ROC-AUC)
    print("\n--- PERFORMANCE METRICS (ROC-AUC) ---")
    print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")
    print(f"Random Forest ROC-AUC:       {roc_auc_score(y_test, rf_probs):.4f}")
    
    # 8. Save Models and Preprocessor
    print("\n--- Saving Artifacts ---")
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr, 'models/logistic_regression_model.pkl')
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print("All models and preprocessor saved to 'models/' directory.")

if __name__ == "__main__":
    run_training_pipeline()
