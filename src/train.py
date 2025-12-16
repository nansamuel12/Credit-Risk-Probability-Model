
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
import os
import sys

# Add src to path to import data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import build_training_pipeline

def train_model(data_path):
    # MLflow Setup
    mlflow.set_experiment("Credit_Risk_Model")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define Target and Features
    target = 'Risk_Label'
    if target not in df.columns:
        raise ValueError(f"Target {target} not found in dataset. Run data processing first.")
        
    X = df.drop(columns=[target, 'CustomerId', 'Cluster', 'FraudResult_max']) # Drop ID and intermediate targets
    y = df[target]
    
    # Identify Column Types for Pipeline
    # Categoricals for WoE/OHE
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or c == 'ProductCategory']
    # Numerics
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    print(f"Categorical Features: {cat_cols}")
    print(f"Numerical Features: {num_cols}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build Preprocessor
    preprocessor = build_training_pipeline(cat_cols, num_cols)
    
    # Full Pipeline
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    # Task 5: Hyperparameter Tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    print("Starting Grid Search...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    
    with mlflow.start_run():
        # Fit
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"Best Params: {best_params}")
        print(f"Best CV AUC: {best_score}")
        
        # Log Params
        mlflow.log_params(best_params)
        
        # Evaluate on Test
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Test AUC: {auc}")
        print(classification_report(y_test, y_pred))
        
        # Log Metrics
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_accuracy", acc)
        
        # Log Model (Task 6 preparation: Model Registry)
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="CreditRiskModel")
        
        # Save locally as fallback
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_risk_model.pkl')
        print("Model saved locally to models/best_risk_model.pkl")

if __name__ == "__main__":
    train_model("data/processed/customer_risk_data.csv")
