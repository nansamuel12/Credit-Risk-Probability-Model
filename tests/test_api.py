
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Create mock for sklearn before importing api
# This is tricky because main imports joblib which loads real models.
# We need to patch joblib.load to return mocks.

with patch('joblib.load') as mock_load:
    # Setup mocks
    mock_risk_model = MagicMock()
    mock_risk_model.predict_proba.return_value = [[0.8, 0.2]] # Low risk
    
    mock_amount_model = MagicMock()
    mock_amount_model.predict.return_value = [5000.0]
    
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = [[0] * 26] # Shape 1, 26
    
    # Side effect: return different mock based on call
    def side_effect(path):
        if 'risk' in str(path): return mock_risk_model
        if 'amount' in str(path): return mock_amount_model
        if 'preprocessor' in str(path): return mock_preprocessor
        return MagicMock()
    
    mock_load.side_effect = side_effect

    # Import app after patching
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk Model API is running."}

# We need to mock the global models in main.py because the app startup loads them
# TestClient triggers startup events.
# However, `joblib.load` was patched during import, but startup runs at runtime.
# So we need to patch `joblib.load` during the test execution too?
# Or easier: Patch the global variables in main directly.

from src.api import main

def test_predict_endpoint():
    # Mock models directly on the module
    main.risk_model = MagicMock()
    main.risk_model.predict_proba.return_value = [[0.9, 0.1]] # 0.1 risk -> Low Risk
    main.risk_model.n_features_in_ = 26
    
    main.amount_model = MagicMock()
    main.amount_model.predict.return_value = [1000.0]
    
    main.preprocessor = MagicMock()
    # Return numpy array
    import numpy as np
    main.preprocessor.transform.return_value = np.zeros((1, 26))
    
    payload = {
        "CustomerId": "TestC1",
        "Total_Transactions": 10,
        "Total_Amount": 5000.0,
        "Average_Amount": 500.0,
        "Amount_std": 50.0,
        "Amount_min": 100.0,
        "Amount_max": 1000.0,
        "Total_Value": 5000.0,
        "Value_mean": 500.0
    }
    
    response = client.post("/predict", json=payload)
    
    # Debug print
    if response.status_code != 200:
        print(response.json())
        
    assert response.status_code == 200
    data = response.json()
    assert data["CustomerId"] == "TestC1"
    assert data["RiskCategory"] == "Low Risk" # 0.1 prob
    assert "CreditScore" in data
    assert "OptimalLoanAmount" in data
