import pandas as pd
import numpy as np
import os
from datetime import datetime

"""
RFM Features Creation (Simple Explanation)
------------------------------------------
RFM = Recency, Frequency, Monetary

This script transforms transaction-level data (many rows per customer) 
into customer-level data (one row per customer) with summary features.

1. Recency: How recently did the customer make a transaction? (today - last transaction date)
2. Frequency: How many transactions did they make? (total count)
3. Monetary: How much do they spend? (total sum of amount)
"""

def create_rfm_features(input_path='data/data.csv', output_path='data/rfm_features.csv'):
    # 1. Load the data
    print(f"--- Step 1: Loading data from {input_path} ---")
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return
    
    df = pd.read_csv(input_path)
    
    # 2. Convert TransactionStartTime to datetime
    print("--- Step 2: Processing dates ---")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # 3. Define 'Today' for Recency calculation
    # We use the most recent transaction date in the dataset as our reference point
    reference_date = df['TransactionStartTime'].max()
    print(f"Reference Date (Latest Transaction): {reference_date}")
    
    # 4. Calculate RFM Features
    print("--- Step 3: Calculating RFM Features ---")
    
    # Group by CustomerId and aggregate
    rfm = df.groupby('CustomerId').agg({
        # Recency: Days since last transaction
        'TransactionStartTime': lambda x: (reference_date - x.max()).days,
        
        # Frequency: Total number of transactions
        'TransactionId': 'count',
        
        # Monetary: Total amount spent
        'Amount': 'sum'
    }).reset_index()
    
    # 5. Rename columns for clarity
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    
    # 6. Show the transformation results
    print("\n--- TRANSFORMATION SUMMARY ---")
    print(f"Before: {len(df)} transaction rows")
    print(f"After: {len(rfm)} customer rows (One row per customer)")
    
    print("\n--- SAMPLE DATA (BEFORE) ---")
    print(df[['CustomerId', 'TransactionStartTime', 'Amount']].head())
    
    print("\n--- SAMPLE DATA (AFTER - RFM) ---")
    print(rfm.head())
    
    # 7. Quality Check: Check for risky patterns
    print("\n--- QUICK RISK INSIGHTS ---")
    print("High Recency (e.g., > 30 days) -> Potential risk (Inactive)")
    print("Low Frequency (e.g., 1 transaction) -> Potential risk (New/Unstable)")
    print("Low Monetary -> Potential risk (Low engagement)")
    
    # 8. Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rfm.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved RFM features to {output_path}")

if __name__ == "__main__":
    create_rfm_features()
