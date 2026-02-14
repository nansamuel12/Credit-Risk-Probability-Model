
import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import CustomerAggregator, add_rf_risk_label
from src.config import ColumnConfig

class TestDataProcessing:
    
    @pytest.fixture
    def sample_transaction_data(self):
        # Create minimal DataFrame for testing
        data = {
            'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'BatchId': ['B1', 'B1', 'B2', 'B2', 'B3'],
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
            'SubscriptionId': ['S1', 'S1', 'S2', 'S2', 'S3'],
            'PricingStrategy': [2, 2, 4, 4, 1], # Categorical
            'Amount': [1000, 500, 200, 300, 5000],
            'Value': [1000, 500, 200, 300, 5000],
            'TransactionStartTime': [
                '2019-01-01T00:00:00Z', 
                '2019-01-02T00:00:00Z', 
                '2018-12-01T00:00:00Z', 
                '2018-12-05T00:00:00Z',
                '2019-01-15T00:00:00Z'
            ],
            'ProductCategory': ['airtime', 'airtime', 'data_bundles', 'data_bundles', 'financial_services'],
            'ChannelId': ['ChannelId_1', 'ChannelId_1', 'ChannelId_2', 'ChannelId_2', 'ChannelId_3'],
            'FraudResult': [0, 0, 0, 0, 1]
        }
        return pd.DataFrame(data)

    def test_customer_aggregator_transform(self, sample_transaction_data):
        aggregator = CustomerAggregator()
        result = aggregator.transform(sample_transaction_data)
        
        # Check output shape (3 unique customers)
        assert result.shape[0] == 3
        
        # Check if customer IDs are correct
        assert 'CustomerId' in result.columns
        assert set(result['CustomerId']) == {'C1', 'C2', 'C3'}
        
        # Check aggregated columns exist
        expected_cols = ['Frequency', 'Monetary', 'Recency', 'PricingStrategy']
        for col in expected_cols:
            assert col in result.columns
            
        # Check specific aggregation logic
        # C1 Total Amount = 1500
        c1_monetary = result.loc[result['CustomerId'] == 'C1', 'Monetary'].values[0]
        assert c1_monetary == 1500
        
        # C2 Frequency = 2
        c2_freq = result.loc[result['CustomerId'] == 'C2', 'Frequency'].values[0]
        assert c2_freq == 2

    def test_add_rf_risk_label(self, sample_transaction_data):
        # First aggregate
        aggregator = CustomerAggregator()
        df_agg = aggregator.transform(sample_transaction_data)
        
        # Then label
        result = add_rf_risk_label(df_agg)
        
        # Check columns
        assert 'Cluster' in result.columns
        assert 'Risk_Label' in result.columns
        
        # Check Risk Label is binary
        assert set(result['Risk_Label'].unique()).issubset({0, 1})

    def test_missing_date_column_handling(self):
        # Dataframe without TransactionStartTime
        df = pd.DataFrame({'CustomerId': ['C1'], 'Amount': [100]})
        aggregator = CustomerAggregator()
        
        # Should gracefully handle or warn (in implementation it prints warning and returns X)
        result = aggregator.transform(df)
        assert result.equals(df)
