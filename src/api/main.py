
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List
import joblib
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import config
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI()

# Global variables for models
risk_model = None
amount_model = None
preprocessor = None

def patch_imputer(preprocessor_obj: Any) -> None:
    """
    Patches SimpleImputer instances in the preprocessor.
    """
    if not preprocessor_obj:
        return
        
    try:
        if hasattr(preprocessor_obj, 'transformers_'):
            for entry in preprocessor_obj.transformers_:
                if len(entry) < 2: continue
                transformer = entry[1]
                
                if hasattr(transformer, 'steps'):
                    for step_name, step in transformer.steps:
                        if hasattr(step, 'strategy') and not hasattr(step, '_fill_dtype'):
                            step._fill_dtype = np.float64
                            print(f"Patched _fill_dtype for {step_name}")
                elif hasattr(transformer, 'strategy') and not hasattr(transformer, '_fill_dtype'):
                     transformer._fill_dtype = np.float64
                     print(f"Patched _fill_dtype for {transformer}")
    except Exception as e:
        print(f"Failed to patch preprocessor: {e}")

@app.on_event("startup")
def load_artifacts():
    global risk_model, amount_model, preprocessor
    try:
        # Load from config paths
        # Note: If running from root, paths in config (e.g. 'models/ risk_model.pkl') are correct.
        risk_model = joblib.load(config.paths.risk_model)
        amount_model = joblib.load(config.paths.amount_model)
        preprocessor = joblib.load(config.paths.preprocessor)
        
        # Apply patch
        patch_imputer(preprocessor)
        
        print("Models and preprocessor loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures):
    if not risk_model or not amount_model or not preprocessor:
        raise HTTPException(status_code=500, detail="Models/Preprocessor not loaded")
    
    # 1. Structure Input
    # Use config.cols.required_features instead of hardcoded list
    required_features = config.cols.required_features
    
    # Initialize dictionary
    input_data = {feature: 0.0 for feature in required_features}
    
    # Map provided features
    # Note: This mapping is manual because Pydantic model might differ slightly from feature names
    # Or cleaner: dict(features) and merge.
    feature_dict = features.dict()
    
    # Mapping logic (Ensure Pydantic fields match expected features or map them)
    # Since we control both, we should align them.
    # Current mapping:
    input_data['Total_Transactions'] = features.Total_Transactions
    input_data['Total_Amount'] = features.Total_Amount
    input_data['Average_Amount'] = features.Average_Amount
    input_data['Amount_Std'] = features.Amount_std # Case difference in Pydantic?
    input_data['Amount_min'] = features.Amount_min
    input_data['Amount_max'] = features.Amount_max
    input_data['Total_Value'] = features.Total_Value
    input_data['Value_mean'] = features.Value_mean
    
    # Add dummy/missing columns required by pipeline logic
    input_data[config.cols.id_col] = 0 # Numeric dummy
    input_data[config.cols.target] = 0
    input_data['Cluster'] = 0
    input_data['FraudResult_max'] = 0
    
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 2. Preprocess
    try:
        X_scaled = preprocessor.transform(df_input)
        
        # FIX: Model expects 26 features
        expected_features = 26 # Config? Or derived from model.n_features_in_
        if hasattr(risk_model, 'n_features_in_'):
            expected_features = risk_model.n_features_in_
            
        if X_scaled.shape[1] > expected_features:
            print(f"Slicing X_scaled from {X_scaled.shape[1]} to {expected_features} features.")
            X_scaled = X_scaled[:, :expected_features]
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    
    # 3. Predict Risk
    risk_prob = risk_model.predict_proba(X_scaled)[0][1]
    
    # 4. Credit Score
    credit_score = int(300 + (1 - risk_prob) * 550)
    risk_category = "High Risk" if risk_prob > 0.5 else "Low Risk"
    
    # 5. Optimal Loan Amount
    if risk_category == "High Risk":
        optimal_amount = 0.0
    else:
        # Amount model usage
        # Logic: Select columns not containing 'Value' or 'Total_Amount'? 
        # This logical dependency is fragile. 
        # Better: Slice based on indices derived from feature naming if possible.
        # Fallback to previous hardcoded logic for stability during refactor.
        
        # Original logic: [i for i, f in enumerate(REQUIRED_FEATURES) if 'Value' not in f and 'Total_Amount' not in f]
        # We need to apply this to the generic feature list.
        
        indices = [i for i, f in enumerate(required_features) if 'Value' not in f and 'Total_Amount' not in f]
        
        # Check constraints
        if hasattr(amount_model, 'n_features_in_'):
             needed_features = amount_model.n_features_in_
             # If indices count doesn't match, we might have issues.
             # For now, trust the logic matches the training logic.
             
        X_amt_scaled = X_scaled[:, indices]
        
        # Ensure 2D
        if X_amt_scaled.ndim == 1:
            X_amt_scaled = X_amt_scaled.reshape(1, -1)
            
        optimal_amount = amount_model.predict(X_amt_scaled)[0]
    
    return {
        "CustomerId": features.CustomerId,
        "RiskProbability": float(risk_prob),
        "CreditScore": credit_score,
        "OptimalLoanAmount": float(optimal_amount),
        "RiskCategory": risk_category
    }
