import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary features.
    Output: One row per customer.
    """
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Reference date (Latest transaction in dataset)
    ref_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        # Recency: Days since last transaction
        'TransactionStartTime': lambda x: (ref_date - x.max()).days,
        # Frequency: Count of transactions
        'TransactionId': 'count',
        # Monetary: Total amount spent
        'Amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def create_proxy_target(df_rfm: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Use KMeans clustering on RFM to create a proxy 'Risk_Label'.
    Risk_Label = 1 for the cluster with the highest average Recency (Inactive customers).
    """
    df_rfm = df_rfm.copy()
    
    # Select RFM for clustering
    features = ['Recency', 'Frequency', 'Monetary']
    X = df_rfm[features]
    
    # Scaling is crucial for KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    df_rfm['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Determine the "High Risk" cluster (Highest Recency = Longest inactivity)
    avg_recency = df_rfm.groupby('Cluster')['Recency'].mean()
    high_risk_cluster = avg_recency.idxmax()
    
    # Assign Risk_Label
    df_rfm['Risk_Label'] = (df_rfm['Cluster'] == high_risk_cluster).astype(int)
    
    print(f"Proxy Target Created. Risk Cluster ID: {high_risk_cluster}")
    print(df_rfm.groupby('Risk_Label')[features].mean())
    
    return df_rfm

if __name__ == "__main__":
    # Test on raw data
    from src.data_processing import load_data
    from src.config import config
    
    raw_df = load_data(str(config.paths.raw_data))
    rfm_df = calculate_rfm(raw_df)
    labeled_df = create_proxy_target(rfm_df)
    print(labeled_df.head())
