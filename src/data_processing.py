import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data folder not found at {path}")
    return pd.read_csv(path)

def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data to customer-level summary features."""
    # Ensure datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Basic Aggregations
    agg_rules = {
        'TransactionId': 'count',
        'Amount': ['sum', 'mean', 'std', 'min', 'max'],
        'Value': ['sum', 'mean']
    }
    
    # Customer Grouping
    df_agg = df.groupby('CustomerId').agg(agg_rules)
    
    # Flatten columns
    df_agg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_agg.columns]
    df_agg = df_agg.reset_index()
    
    # Handle NaNs from std
    df_agg.fillna(0, inplace=True)
    
    # Rename for clarity
    df_agg.rename(columns={
        'TransactionId_count': 'Total_Transactions',
        'Amount_sum': 'Total_Amount',
        'Amount_mean': 'Average_Amount'
    }, inplace=True)
    
    return df_agg

def preprocessing_pipeline(num_cols: list, cat_cols: list = None) -> ColumnTransformer:
    """Build a scikit-learn preprocessing pipeline."""
    
    # Numeric Transformer
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    transformers = [('num', num_transformer, num_cols)]
    
    # Optional Categorical Transformer
    if cat_cols:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, cat_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

if __name__ == "__main__":
    # Example usage/test
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.config import config
    data = load_data(str(config.paths.raw_data))
    agg_data = aggregate_customer_features(data)
    print(f"Aggregated Data Shape: {agg_data.shape}")
    print(agg_data.head())
