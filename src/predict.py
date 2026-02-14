
import pandas as pd
import joblib
import os
import mlflow.sklearn

def load_model(model_path='models/best_risk_model.pkl'):
    """
    Loads model. Prioritizes MLflow Registry if MLFLOW_TRACKING_URI is set.
    Otherwise falls back to local pickle.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_model_name = os.getenv("MLFLOW_MODEL_NAME", "CreditRiskModel_RF") 
    mlflow_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    
    if mlflow_uri:
        try:
            print(f"MLflow URI detected: {mlflow_uri}")
            # Load from Registry
            model_uri = f"models:/{mlflow_model_name}/{mlflow_stage}"
            print(f"Attempting to load from {model_uri}...")
            model = mlflow.sklearn.load_model(model_uri)
            print("Successfully loaded model from MLflow Registry")
            return model
        except Exception as e:
            print(f"Failed to load from MLflow registry: {e}. Checking for latest version...")
            # Try latest if Production fails
            try:
                model_uri = f"models:/{mlflow_model_name}/None"
                model = mlflow.sklearn.load_model(model_uri)
                print("Loaded latest version from MLflow Registry")
                return model
            except Exception as e2:
                 print(f"Failed to load from MLflow completely: {e2}. Falling back to local.")

    if os.path.exists(model_path):
        print(f"Loading local model from {model_path}")
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"No model found at {model_path} and MLflow load failed/skipped.")

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
