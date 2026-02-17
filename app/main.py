"""
FastAPI application for Credit Risk Probability Prediction
Uses MLflow model registry to load production models
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Probability API",
    description="API for predicting credit risk probability using RFM features",
    version="1.0.0"
)

# Global model cache
model_cache = {
    "model": None,
    "preprocessor": None,
    "model_version": None,
    "preprocessor_version": None
}


class RFMFeatures(BaseModel):
    """Input schema for RFM features"""
    Recency: float = Field(..., description="Days since last transaction", ge=0)
    Frequency: float = Field(..., description="Total number of transactions", ge=1)
    Monetary: float = Field(..., description="Total amount spent", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "Recency": 30,
                "Frequency": 15,
                "Monetary": 5000.0
            }
        }


class PredictionRequest(BaseModel):
    """Request schema for single or batch predictions"""
    features: List[RFMFeatures] = Field(..., description="List of RFM features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {
                        "Recency": 30,
                        "Frequency": 15,
                        "Monetary": 5000.0
                    },
                    {
                        "Recency": 120,
                        "Frequency": 3,
                        "Monetary": 500.0
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predictions: List[Dict[str, float]]
    model_name: str
    model_version: str
    preprocessor_version: str
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "risk_probability": 0.15,
                        "low_risk_probability": 0.85,
                        "risk_label": 0
                    },
                    {
                        "risk_probability": 0.82,
                        "low_risk_probability": 0.18,
                        "risk_label": 1
                    }
                ],
                "model_name": "credit_risk_randomforest",
                "model_version": "1",
                "preprocessor_version": "1"
            }
        }


def load_production_model():
    """Load the production model from MLflow Model Registry"""
    try:
        client = MlflowClient()
        
        # Get production models
        model_name = None
        for registered_model in client.search_registered_models():
            if registered_model.name.startswith("credit_risk") and \
               registered_model.name != "credit_risk_preprocessor":
                model_name = registered_model.name
                break
        
        if not model_name:
            raise ValueError("No credit risk model found in registry")
        
        # Load model from Production stage
        model_uri = f"models:/{model_name}/Production"
        preprocessor_uri = "models:/credit_risk_preprocessor/Production"
        
        logger.info(f"Loading model: {model_uri}")
        logger.info(f"Loading preprocessor: {preprocessor_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        preprocessor = mlflow.pyfunc.load_model(preprocessor_uri)
        
        # Get version information
        model_versions = client.get_latest_versions(model_name, stages=["Production"])
        preprocessor_versions = client.get_latest_versions(
            "credit_risk_preprocessor", 
            stages=["Production"]
        )
        
        model_version = model_versions[0].version if model_versions else "Unknown"
        preprocessor_version = preprocessor_versions[0].version if preprocessor_versions else "Unknown"
        
        return model, preprocessor, model_name, model_version, preprocessor_version
        
    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        logger.info("Attempting to load from local models directory...")
        
        # Fallback to local models
        import joblib
        models_dir = "models"
        
        # Try to find the best model
        rf_path = os.path.join(models_dir, "random_forest_model.pkl")
        lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        
        if os.path.exists(rf_path):
            model = joblib.load(rf_path)
            model_name = "credit_risk_randomforest_local"
        elif os.path.exists(lr_path):
            model = joblib.load(lr_path)
            model_name = "credit_risk_logisticregression_local"
        else:
            raise FileNotFoundError("No trained models found")
        
        preprocessor = joblib.load(preprocessor_path)
        
        logger.info(f"Loaded model from local directory: {model_name}")
        
        return model, preprocessor, model_name, "local", "local"


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading production models...")
    try:
        model, preprocessor, model_name, model_version, preprocessor_version = load_production_model()
        
        model_cache["model"] = model
        model_cache["preprocessor"] = preprocessor
        model_cache["model_name"] = model_name
        model_cache["model_version"] = model_version
        model_cache["preprocessor_version"] = preprocessor_version
        
        logger.info(f"✓ Model loaded: {model_name} v{model_version}")
        logger.info(f"✓ Preprocessor loaded: v{preprocessor_version}")
        logger.info("API ready to serve predictions")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Probability Prediction API",
        "version": "1.0.0",
        "model": model_cache.get("model_name", "Not loaded"),
        "model_version": model_cache.get("model_version", "N/A"),
        "status": "online" if model_cache["model"] is not None else "model not loaded"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_cache["model"] is None or model_cache["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_cache["model_name"],
        "model_version": model_cache["model_version"],
        "preprocessor_version": model_cache["preprocessor_version"]
    }


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_cache["model_name"],
        "model_version": model_cache["model_version"],
        "preprocessor_version": model_cache["preprocessor_version"],
        "stage": "Production",
        "features": ["Recency", "Frequency", "Monetary"],
        "output": "Risk probability (0 = Low Risk, 1 = High Risk)"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk probability for given RFM features
    
    Returns:
        - risk_probability: Probability of being high risk (0-1)
        - low_risk_probability: Probability of being low risk (0-1)
        - risk_label: Binary prediction (0=Low Risk, 1=High Risk)
    """
    if model_cache["model"] is None or model_cache["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([
            {
                "Recency": feat.Recency,
                "Frequency": feat.Frequency,
                "Monetary": feat.Monetary
            }
            for feat in request.features
        ])
        
        logger.info(f"Received prediction request for {len(input_data)} samples")
        
        # Preprocess features
        # For MLflow models, we need to handle this differently
        if hasattr(model_cache["preprocessor"], "predict"):
            # MLflow wrapped model
            # Since preprocessor is a transformer, we need to use the original sklearn model
            import joblib
            preprocessor_sklearn = joblib.load("models/preprocessor.pkl")
            X_scaled = preprocessor_sklearn.transform(input_data)
        else:
            X_scaled = model_cache["preprocessor"].transform(input_data)
        
        # Make predictions
        if hasattr(model_cache["model"], "predict_proba"):
            # Direct sklearn model
            probabilities = model_cache["model"].predict_proba(X_scaled)
            predictions_binary = model_cache["model"].predict(X_scaled)
        else:
            # MLflow wrapped model - load original sklearn model
            import joblib
            # Determine which model to use
            if "randomforest" in model_cache["model_name"].lower():
                model_sklearn = joblib.load("models/random_forest_model.pkl")
            else:
                model_sklearn = joblib.load("models/logistic_regression_model.pkl")
            
            probabilities = model_sklearn.predict_proba(X_scaled)
            predictions_binary = model_sklearn.predict(X_scaled)
        
        # Format response
        predictions = []
        for i in range(len(input_data)):
            predictions.append({
                "risk_probability": float(probabilities[i, 1]),  # Probability of class 1 (High Risk)
                "low_risk_probability": float(probabilities[i, 0]),  # Probability of class 0 (Low Risk)
                "risk_label": int(predictions_binary[i])
            })
        
        logger.info(f"Predictions completed successfully")
        
        return PredictionResponse(
            predictions=predictions,
            model_name=model_cache["model_name"],
            model_version=str(model_cache["model_version"]),
            preprocessor_version=str(model_cache["preprocessor_version"])
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/single")
async def predict_single(features: RFMFeatures):
    """
    Predict credit risk probability for a single customer
    
    Convenience endpoint for single predictions
    """
    request = PredictionRequest(features=[features])
    response = await predict(request)
    
    return {
        "prediction": response.predictions[0],
        "model_name": response.model_name,
        "model_version": response.model_version
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
