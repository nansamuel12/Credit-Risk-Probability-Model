

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score
import joblib
import os


def train_model(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Select Features (Dynamic)
    exclude_cols = ['CustomerId', 'Risk_Label']
    features = [c for c in df.columns if c not in exclude_cols]
    print(f"Training with {len(features)} features: {features}")
    
    target_risk = 'Risk_Label'
    if target_risk not in df.columns:
         # Fallback if old file
         target_risk = 'Risk_Label_Binary'
    
    X = df[features]
    y_risk = df[target_risk]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
    
    # --- Task 3: Risk Probability Model ---
    print("\nTraining Risk Model...")
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    risk_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = risk_model.predict(X_test)
    y_prob = risk_model.predict_proba(X_test)[:, 1]
    print("Risk Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob)}")
    
    # Feature Importance (Task 2)
    importances = pd.Series(risk_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importances (Risk Predictors):")
    print(importances)
    
    # --- Task 4: Credit Score Model ---
    # We don't train a model *for* the score, we derive it from the Risk Probability.
    # Score = (1 - RiskProb) * 850 (Standard Scale usually 300-850)
    # Let's define a function for it in predict.py later.
    
    # --- Task 5: Optimal Amount Model (Regression) ---
    # We want to predict 'Total_Value' (or similar capacity metric) for Good Customers.
    # We'll assume 'Total_Value' correlates with what they can handle.
    # Training only on Low Risk customers.
    
    print("\nTraining Amount Limit Model (on non-fraud/low-risk customers)...")
    good_customers = df[df[target_risk] == 0]
    X_good = good_customers[features].drop(columns=['Total_Value', 'Total_Amount']) # Don't leak target if target is related
    # Actually, predicting "Total_Value" using "Total_Transactions" and "Average_Amount" is creating a circular dependency if not careful.
    # The prompt asks for "Optimal amount".
    # Let's predict 'Value_mean' (Average transaction value) as a proxy for "Loan Size" they are comfortable with?
    # Or 'Total_Value'?

    # Let's predict 'Value_mean' (Average transaction value) as a proxy
    # We need to make sure we use the SCALED target if everything is scaled?
    # Actually, the target `Value_mean` is also scaled in the CSV!
    # If we predict scaled value, we must unscale it for the user.
    # But `StandardScaler` scales ALL numericals including 'Value_mean'.
    # This is a bit tricky for the "Optimal Amount" output in API.
    # We will predict the scaled value, and the API output will be a "Score".
    # Or we can inverse transform... but the Pipeline transforms everything together.
    # For now, let's predict the scaled 'Value_mean'.
    
    target_amount = 'Value_mean'
    
    # Use features that are good predictors but not the target itself (Total_Value, Value_mean)
    # Filter features list
    features_amount = [f for f in features if 'Value' not in f and 'Total_Amount' not in f]
    print(f"Features for Amount Model: {features_amount}")
    
    X_amt = good_customers[features_amount]
    y_amt = good_customers[target_amount]
    
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_amt, y_amt, test_size=0.2, random_state=42)
    
    amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
    amount_model.fit(X_train_a, y_train_a)
    
    y_pred_a = amount_model.predict(X_test_a)
    mse = mean_squared_error(y_test_a, y_pred_a)
    print(f"Amount Model MSE: {mse}")
    
    # Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump(risk_model, 'models/risk_model.pkl')
    joblib.dump(amount_model, 'models/amount_model.pkl')
    print("\nModels saved to models/")


if __name__ == "__main__":
    train_model("data/processed/customer_risk_data_pipeline.csv")
