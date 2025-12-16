
import pytest
import pandas as pd
import numpy as np
from src.data_processing import CustomerAggregator, DateFeatureExtractor, build_training_pipeline

def test_date_extractor():
    df = pd.DataFrame({'TransactionStartTime': ['2023-01-01 10:00:00']})
    extractor = DateFeatureExtractor()
    res = extractor.transform(df)
    assert 'TransactionHour' in res.columns
    assert res['TransactionHour'].iloc[0] == 10

def test_customer_aggregator_columns():
    # Mock Data
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 50],
        'Value': [100, 200, 50],
        'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01']),
        'ChannelId': ['Web', 'Web', 'Mobile'],
        'ProductCategory': ['A', 'B', 'A'],
        'PricingStrategy': ['1', '1', '2'],
        'ProviderId': ['P1', 'P1', 'P2'],
        'FraudResult': [0, 0, 1]
    })
    
    agg = CustomerAggregator()
    res = agg.transform(df)
    
    expected_cols = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'FraudResult_max']
    for col in expected_cols:
        assert col in res.columns
        
    assert res.shape[0] == 2 # 2 Customers
    assert res[res['CustomerId'] == 'C1']['Frequency'].iloc[0] == 2

def test_pipeline_build():
    preprocessor = build_training_pipeline(['ChannelId'], ['Amount'])
    assert preprocessor is not None
