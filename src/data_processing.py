
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import category_encoders as ce
import os
import joblib

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts date features from TransactionStartTime"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X['TransactionStartTime']):
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data to customer level.
    Computes RFM (Recency, Frequency, Monetary).
    Keeps categorical modes for downstream encoding.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure Date Features
        if 'TransactionYear' not in X.columns:
            extractor = DateFeatureExtractor()
            X = extractor.transform(X)
            
        # RFM Calculation
        # Recency: Days since last transaction
        # We need a reference point. Max date in dataset?
        ref_date = X['TransactionStartTime'].max()
        
        # Categorical Modes (for WoE/OHE later)
        # We'll take the most frequent Channel, Product, etc.
        cat_cols = ['ChannelId', 'ProductCategory', 'PricingStrategy', 'ProviderId']
        existing_cats = [c for c in cat_cols if c in X.columns]
        
        agg_rules = {
            'TransactionId': 'count', # Frequency
            'Amount': ['sum', 'mean', 'std', 'min', 'max'], # Monetary
            'Value': ['sum', 'mean'],
            'TransactionStartTime': lambda x: (ref_date - x.max()).days, # Recency
            'FraudResult': 'max' # Existing label
        }
        
        # Add categorical mode aggregation
        # Lambda for mode is slow, but usually fine for reasonable datasets
        for cat in existing_cats:
            agg_rules[cat] = lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'

        # Groupby
        customer_stats = X.groupby('CustomerId').agg(agg_rules)
        
        # Flatten MultiIndex
        new_cols = []
        for col in customer_stats.columns:
            if isinstance(col, tuple):
                if col[0] == 'TransactionId' and col[1] == 'count':
                    new_cols.append('Frequency')
                elif col[0] == 'TransactionStartTime' and '<lambda>' in str(col[1]):
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
        
        # Fill NaNs (Std might be NaN if only 1 transaction)
        customer_stats.fillna(0, inplace=True) 

        return customer_stats

def add_rf_risk_label(df_agg):
    """
    Performs K-Means clustering on RFM to create a proxy risk label.
    Task 4 compliance.
    """
    # Select RFM columns
    rfm = df_agg[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Scale for Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # K-Means k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_agg['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify Risk Cluster
    # Assumption: High Recency (Inactive) or Low Monetary/Frequency is 'Riskier' for credit?
    # OR: Maybe High Frequency + High Monetary with Fraud history?
    # Let's align with 'FraudResult' if possible.
    # Check mean FraudResult per cluster
    cluster_risk = df_agg.groupby('Cluster')['FraudResult_max'].mean()
    
    # If FraudResult is too sparse, use RFM logic:
    # High Risk = Cluster with highest Recency (Not transacting) or Lowest Monetary
    # Let's define High Risk as the cluster with Highest Recency (Chur/Bad Credit Risk)
    # The user manual check would be ideal, but for automation:
    avg_recency = df_agg.groupby('Cluster')['Recency'].mean()
    high_risk_cluster = avg_recency.idxmax()
    
    # Create Binary Label
    df_agg['Risk_Label'] = df_agg['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
    
    print(f"Clusters defined. Risk Cluster (High Recency): {high_risk_cluster}")
    print(df_agg.groupby('Risk_Label')[['Recency', 'Frequency', 'Monetary']].mean())
    
    return df_agg

def build_training_pipeline(cat_cols, num_cols):
    """
    Task 3: Pipeline with WoE and OHE
    """
    # Categorical Pipeline: WoE -> OHE (or just WoE if high cardinality)
    # The prompt asks for WoE AND OHE. 
    # Usually you pick one. We will split:
    # High cardinality (ProviderId?) -> WoE
    # Low cardinality (ProductCategory?) -> OHE
    
    # Implementation: Apply WoE to all, then OHE? No, WoE makes it numeric.
    # Let's apply OHE to 'ProductCategory' and WoE to others.
    
    # For simplicity and robust compliance:
    # We will use WoE for all categoricals as it handles new categories well and encodes info.
    # To strictly satisfy "integrate categorical encoding via sklearn (e.g., OneHotEncoder...)", 
    # we will OHE 'ProductCategory' specifically.
    
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
             ('woe', ce.WOEEncoder()) # Requires target y in fit
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
        remainder='drop' # Drop ID, etc.
    )
    
    return preprocessor

def process_data(input_path, output_path):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Aggregation & Feature Creation
    # Note: We do this OUTSIDE the sklearn pipeline fit/transform usually, 
    # because it reduces rows (N -> M).
    print("Aggregating to Customer Level...")
    aggregator = CustomerAggregator()
    df_agg = aggregator.transform(df)
    
    # 2. Target Engineering
    print("Engineering Proxy Target (RFM + KMeans)...")
    df_labeled = add_rf_risk_label(df_agg)
    
    # Save processed (but not yet scaled/encoded) data for Training
    # The Training script will use the Pipeline to encode/scale.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_labeled.to_csv(output_path, index=False)
    print(f"Saved customer-level data to {output_path}")
    print(df_labeled.head())
    
    return df_labeled

if __name__ == "__main__":
    process_data('data/raw/data.csv', 'data/processed/customer_risk_data.csv')
