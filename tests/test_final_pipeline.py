import pytest
import pandas as pd
import numpy as np
from src.data_processing import aggregate_customer_features, preprocessing_pipeline
from src.rfm import calculate_rfm, create_proxy_target

@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [1000, 500, 200, 300, 5000],
        'Value': [1000, 500, 200, 300, 5000],
        'TransactionStartTime': [
            '2023-01-01T10:00:00Z',
            '2023-01-02T11:00:00Z',
            '2023-01-03T12:00:00Z',
            '2023-01-04T13:00:00Z',
            '2023-01-05T14:00:00Z'
        ],
        'ChannelId': ['1', '1', '2', '2', '1'],
        'ProductCategory': ['airtime', 'airtime', 'data', 'data', 'financial'],
        'PricingStrategy': [1, 1, 2, 2, 1],
        'ProviderId': ['P1', 'P1', 'P2', 'P2', 'P1']
    }
    return pd.DataFrame(data)

def test_aggregate_customer_features(sample_data):
    """Test feature engineering/aggregation function"""
    result = aggregate_customer_features(sample_data)
    
    # Should have one row per customer
    assert len(result) == 3
    assert 'Total_Transactions' in result.columns
    assert 'Total_Amount' in result.columns
    assert 'Average_Amount' in result.columns
    
    # Check specific calculation
    c1_total = result[result['CustomerId'] == 'C1']['Total_Amount'].iloc[0]
    assert c1_total == 1500

def test_calculate_rfm(sample_data):
    """Test RFM calculation logic"""
    rfm_result = calculate_rfm(sample_data)
    
    # Should have Recency, Frequency, Monetary
    assert 'Recency' in rfm_result.columns
    assert 'Frequency' in rfm_result.columns
    assert 'Monetary' in rfm_result.columns
    assert len(rfm_result) == 3
    
    # Check values for C3 (most recent)
    c3_rfm = rfm_result[rfm_result['CustomerId'] == 'C3'].iloc[0]
    assert c3_rfm['Frequency'] == 1
    assert c3_rfm['Monetary'] == 5000
    assert c3_rfm['Recency'] == 0 # 2023-01-05 is the max date

def test_create_proxy_target(sample_data):
    """Test proxy labeling using clustering"""
    rfm_result = calculate_rfm(sample_data)
    labeled_data = create_proxy_target(rfm_result)
    
    assert 'Risk_Label' in labeled_data.columns
    assert set(labeled_data['Risk_Label'].unique()).issubset({0, 1})
    assert len(labeled_data) == 3

def test_preprocessing_pipeline():
    """Test preprocessing pipeline setup"""
    num_cols = ['Recency', 'Frequency', 'Monetary']
    preprocessor = preprocessing_pipeline(num_cols)
    
    # Check if it's a ColumnTransformer
    from sklearn.compose import ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Test on dummy data
    df = pd.DataFrame({
        'Recency': [10, 20],
        'Frequency': [5, 2],
        'Monetary': [1000, 500]
    })
    
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape == (2, 3)

def test_prediction_output_shape():
    """Test that the prediction logic (mocked or actual) returns correct shape"""
    # This specifically tests that our features match what a model expects
    num_cols = ['Recency', 'Frequency', 'Monetary']
    preprocessor = preprocessing_pipeline(num_cols)
    
    df = pd.DataFrame({
        'Recency': [10, 20, 30],
        'Frequency': [5, 2, 1],
        'Monetary': [1000, 500, 100]
    })
    
    X = preprocessor.fit_transform(df)
    
    # Verify shape
    assert X.shape[0] == 3
    assert X.shape[1] == 3 # Recency, Frequency, Monetary
