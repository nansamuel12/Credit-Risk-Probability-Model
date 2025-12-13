


from fastapi import FastAPI, HTTPException
from .pydantic_models import CustomerFeatures, PredictionResponse
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load models & preprocessor
try:
    risk_model = joblib.load('models/risk_model.pkl')
    amount_model = joblib.load('models/amount_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
except Exception as e:
    print(f"Error loading artifacts: {e}")
    risk_model = None
    amount_model = None
    preprocessor = None

# Features expected by the preprocessor (Unscaled Aggregated Data)
# Note: The preprocessor expects these columns to exist in the input DataFrame
REQUIRED_FEATURES = [
    'Total_Transactions', 'Total_Amount', 'Average_Amount', 'Amount_Std', 'Amount_min', 'Amount_max', 
    'Total_Value', 'Value_mean', 'TransactionHour_mean', 
    'ChannelId_ChannelId_1_sum', 'ChannelId_ChannelId_2_sum', 'ChannelId_ChannelId_3_sum', 'ChannelId_ChannelId_5_sum', 
    'ProductCategory_airtime_sum', 'ProductCategory_data_bundles_sum', 'ProductCategory_financial_services_sum', 
    'ProductCategory_movies_sum', 'ProductCategory_other_sum', 'ProductCategory_ticket_sum', 'ProductCategory_transport_sum', 
    'ProductCategory_tv_sum', 'ProductCategory_utility_bill_sum', 
    'PricingStrategy_0_sum', 'PricingStrategy_1_sum', 'PricingStrategy_2_sum', 'PricingStrategy_4_sum'
]

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures):
    if not risk_model or not amount_model or not preprocessor:
        raise HTTPException(status_code=500, detail="Models/Preprocessor not loaded")
    
    # 1. Structure Input (Default missing stats to 0)
    input_data = {feature: 0.0 for feature in REQUIRED_FEATURES}
    
    # Map provided features
    input_data['Total_Transactions'] = features.Total_Transactions
    input_data['Total_Amount'] = features.Total_Amount
    input_data['Average_Amount'] = features.Average_Amount
    input_data['Amount_Std'] = features.Amount_std
    input_data['Amount_min'] = features.Amount_min
    input_data['Amount_max'] = features.Amount_max
    input_data['Total_Value'] = features.Total_Value
    input_data['Value_mean'] = features.Value_mean
    
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 2. Preprocess (Scale/Impute)
    try:
        X_scaled = preprocessor.transform(df_input)
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
        # Amount model uses subset of features? 
        # In train.py: features_amount = [f for f in features if 'Value' not in f and 'Total_Amount' not in f]
        # But the model.predict expects the SAME shape if it was trained on a slice?
        # WAIT: In train.py, `X_amt` was a subset of columns.
        # `amount_model.fit(X_train_a, ...)`
        # If I pass `X_scaled` (26 cols) to `amount_model.predict`, it will fail if it expects fewer cols.
        
        # We need to slice X_scaled to the columns expected by amount_model.
        # Or easier: amount_model was trained on a DataFrame selection from the processed data.
        # But `X_scaled` is a numpy array (output of Pipeline).
        # It loses column names!
        # This is a common pain point.
        
        # Solution: Re-convert X_scaled to DataFrame to select columns?
        # Or just select indices.
        # Robust way: Get feature names from preprocessor?
        # Alternative: Retrain amount_model on ALL features?
        # For now, let's index the array.
        
        # Map indices
        # features_amount_names = [f for f in REQUIRED_FEATURES if 'Value' not in f and 'Total_Amount' not in f]
        # This requires reconstructing the logic perfectly.
        
        # Index lookup
        indices = [i for i, f in enumerate(REQUIRED_FEATURES) if 'Value' not in f and 'Total_Amount' not in f]
        X_amt_scaled = X_scaled[:, indices]
        
        optimal_amount = amount_model.predict(X_amt_scaled)[0]
    
    return {
        "CustomerId": features.CustomerId,
        "RiskProbability": float(risk_prob),
        "CreditScore": credit_score,
        "OptimalLoanAmount": float(optimal_amount),
        "RiskCategory": risk_category
    }
