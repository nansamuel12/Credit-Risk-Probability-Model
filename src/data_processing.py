
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import category_encoders as ce
import os
import sys
from typing import List, Union, Optional

# Add src to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import config

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts date features from TransactionStartTime"""
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DateFeatureExtractor':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        date_col = config.cols.date_col
        if date_col in X.columns and not pd.api.types.is_datetime64_any_dtype(X[date_col]):
            X[date_col] = pd.to_datetime(X[date_col])
        
        if date_col in X.columns:
            X['TransactionHour'] = X[date_col].dt.hour
            X['TransactionDay'] = X[date_col].dt.day
            X['TransactionMonth'] = X[date_col].dt.month
            X['TransactionYear'] = X[date_col].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data to customer level.
    Computes RFM (Recency, Frequency, Monetary).
    Keeps categorical modes for downstream encoding.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CustomerAggregator':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ensure Date Features
        if 'TransactionYear' not in X.columns:
            extractor = DateFeatureExtractor()
            X = extractor.transform(X)
            
        # RFM Calculation
        date_col = config.cols.date_col
        if date_col not in X.columns:
             # Handle case where date_col is missing
             print(f"Warning: {date_col} missing for RFM calculation.")
             return X
             
        # Recency: Days since last transaction
        ref_date = X[date_col].max()
        
        # Categorical Modes
        cat_cols = config.cols.cat_cols
        existing_cats = [c for c in cat_cols if c in X.columns]
        
        agg_rules = {
            'TransactionId': 'count', # Frequency
            'Amount': ['sum', 'mean', 'std', 'min', 'max'], # Monetary
            'Value': ['sum', 'mean'],
            date_col: lambda x: (ref_date - x.max()).days, # Recency
            config.cols.fraud_label: 'max' # Existing label
        }
        
        # Add categorical mode aggregation
        for cat in existing_cats:
            agg_rules[cat] = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'

        # Groupby
        customer_stats = X.groupby(config.cols.id_col).agg(agg_rules)
        
        # Flatten MultiIndex
        new_cols = []
        for col in customer_stats.columns:
            if isinstance(col, tuple):
                if col[0] == 'TransactionId' and col[1] == 'count':
                    new_cols.append('Frequency')
                elif col[0] == date_col and '<lambda>' in str(col[1]):
                    new_cols.append('Recency')
                elif col[0] == 'Amount' and col[1] == 'sum':
                    new_cols.append('Monetary')
                elif col[1] == '<lambda>':
                    new_cols.append(col[0]) # Keep original name for categorical mode
                else:
                    new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(col)
                
        customer_stats.columns = new_cols
        
        # Reset index
        customer_stats.reset_index(inplace=True)
        
        # Fill NaNs
        customer_stats.fillna(0, inplace=True) 

        return customer_stats

def add_rf_risk_label(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Performs K-Means clustering on RFM to create a proxy risk label.
    Default Logic (Proxy):
    - Identify patterns of behavior (Recency, Frequency, Monetary).
    - High Recency (Inactive) or Low Monetary/Frequency -> Riskier.
    """
    # Select RFM columns
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    if not all(col in df_agg.columns for col in rfm_cols):
        print(f"Warning: Missing RFM columns {rfm_cols}. Cannot create risk label.")
        return df_agg

    rfm = df_agg[rfm_cols].copy()
    
    # Scale for Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # K-Means
    kmeans = KMeans(n_clusters=config.model.n_clusters_risk, random_state=config.model.random_state)
    df_agg['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify Risk Cluster
    # We define High Risk (1) as the cluster with Highest Recency (Inactive)
    # This implies they stopped using the platform or defaulted on engagement.
    # Alternatively, could be Low Monetary.
    # For credit checks: Inactivity is often a sign of churn/default if balance > 0 (but we don't have balance).
    
    avg_recency = df_agg.groupby('Cluster')['Recency'].mean()
    high_risk_cluster = avg_recency.idxmax()
    
    # Create Binary Label
    target_col = config.cols.target
    df_agg[target_col] = df_agg['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
    
    print(f"Clusters defined. Risk Cluster (High Recency): {high_risk_cluster}")
    print(df_agg.groupby(target_col)[rfm_cols].mean())
    
    return df_agg

def build_training_pipeline(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """
    Builds the sklearn preprocessing pipeline with WoE and OHE.
    """
    # Split categoricals for different encoding strategies
    woe_cats = [c for c in cat_cols if c != 'ProductCategory']
    ohe_cats = ['ProductCategory'] if 'ProductCategory' in cat_cols else []
    
    transformers = []
    
    # Numeric: Impute -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    transformers.append(('num', numeric_transformer, num_cols))
    
    # WoE
    if woe_cats:
        woe_transformer = Pipeline(steps=[
             ('imputer', SimpleImputer(strategy='most_frequent')),
             ('woe', ce.WOEEncoder()) 
        ])
        transformers.append(('woe', woe_transformer, woe_cats))
        
    # OHE
    if ohe_cats:
        ohe_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('ohe', ohe_transformer, ohe_cats))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' 
    )
    
    return preprocessor

def process_data(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Aggregation & Feature Creation
    print("Aggregating to Customer Level...")
    aggregator = CustomerAggregator()
    df_agg = aggregator.transform(df)
    
    # 2. Target Engineering
    print("Engineering Proxy Target (RFM + KMeans)...")
    df_labeled = add_rf_risk_label(df_agg)
    
    # Save 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    print(f"Saved customer-level data to {output_path}")
    
    return df_labeled

if __name__ == "__main__":
    process_data(str(config.paths.raw_data), str(config.paths.processed_data))
