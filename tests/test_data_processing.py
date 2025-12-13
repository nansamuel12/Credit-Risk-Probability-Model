

import pytest
import pandas as pd
from src.data_processing import clean_data

def test_clean_data():
    data = {
        'TransactionStartTime': ['2018-11-15T02:18:49Z'],
        'Amount': [1000],
        'Value': [1000], 
        'FraudResult': [0]
    }
    df = pd.DataFrame(data)
    df_cleaned = clean_data(df)
    
    assert 'TransactionHour' in df_cleaned.columns
    assert df_cleaned['TransactionYear'][0] == 2018
