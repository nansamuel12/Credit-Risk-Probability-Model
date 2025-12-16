
import pandas as pd
import joblib
import os
import mlflow.sklearn

def load_model(model_path='models/best_risk_model.pkl', use_mlflow=False):
    if use_mlflow:
        try:
            # Load from Registry (Production alias preferred, or specific version)
            # Assuming 'CreditRiskModel' is the registered name
            model = mlflow.sklearn.load_model("models:/CreditRiskModel/None") # None stage for now, or use Latest
            print("Loaded model from MLflow Registry")
            return model
        except Exception as e:
            print(f"Failed to load from MLflow: {e}. Falling back to local.")
    
    if os.path.exists(model_path):
        print(f"Loading local model from {model_path}")
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"No model found at {model_path}")

def make_prediction(model, input_data: dict):
    """
    Args:
        model: Loaded sklearn pipeline
        input_data: Dictionary of features (e.g. {'Recency': 10, 'Frequency': 5, ...})
    """
    df = pd.DataFrame([input_data])
    
    # Predict Probability
    # The pipeline handles scaling/encoding
    try:
        prob = model.predict_proba(df)[:, 1][0]
        # Credit Score (Task 4 requirement: Assign score based on probability)
        # Simple formula: 300 + (1 - prob) * 550  (Range 300-850)
        credit_score = 300 + (1 - prob) * 550
        
        return {
            "risk_probability": float(prob),
            "credit_score": int(credit_score),
            "is_high_risk": bool(prob > 0.5)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test run
    # Mock data compatible with our new pipeline features
    mock_data = {
        'Recency': 10, 
        'Frequency': 5, 
        'Monetary': 5000, 
        'TransactionHour': 12, 
        'TransactionDay': 15,
        'TransactionMonth': 5,
        'TransactionYear': 2023,
        'ChannelId': 'Web',
        'ProductCategory': 'financial_services',
        'PricingStrategy': '2',
        'ProviderId': 'Provider_1'
    }
    
    try:
        model = load_model()
        result = make_prediction(model, mock_data)
        print(result)
    except Exception as e:
        print(f"Could not run prediction: {e}")
