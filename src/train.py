
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
import os
import sys

# Add src to path to import data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import build_training_pipeline

def evaluate_and_log(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Log to MLflow (assuming active run)
    mlflow.log_metric(f"{model_name}_test_auc", auc)
    mlflow.log_metric(f"{model_name}_test_accuracy", acc)
    mlflow.log_metric(f"{model_name}_test_precision", prec)
    mlflow.log_metric(f"{model_name}_test_recall", rec)
    mlflow.log_metric(f"{model_name}_test_f1", f1)
    
    return auc

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
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or c == 'ProductCategory']
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build Preprocessor
    preprocessor = build_training_pipeline(cat_cols, num_cols)
    
    # --- Experiment 1: Random Forest (Grid Search) ---
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        pipeline_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', rf)
        ])
        
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5]
        }
        
        print("\nStarting Grid Search for RandomForest...")
        grid_search = GridSearchCV(pipeline_rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        print(f"Best RF Params: {grid_search.best_params_}")
        
        mlflow.log_params(grid_search.best_params_)
        evaluate_and_log(best_rf, X_test, y_test, "RandomForest")
        
        # Log Model
        mlflow.sklearn.log_model(best_rf, "model_rf", registered_model_name="CreditRiskModel_RF")
        
        # Save locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_rf, 'models/best_risk_model.pkl')
        
    # --- Experiment 2: Logistic Regression ---
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        pipeline_lr = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', lr)
        ])
        
        print("\nTraining Logistic Regression...")
        pipeline_lr.fit(X_train, y_train)
        
        evaluate_and_log(pipeline_lr, X_test, y_test, "LogisticRegression")
        
        # Log Model
        mlflow.sklearn.log_model(pipeline_lr, "model_lr")
        joblib.dump(pipeline_lr, 'models/logistic_model.pkl')

    print("\nTraining Complete. Models saved.")

if __name__ == "__main__":
    train_model("data/processed/customer_risk_data.csv")
