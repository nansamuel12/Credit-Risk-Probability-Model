import pandas as pd
import os

data_path = 'data/data.csv'
print(f"--- Loading {data_path} ---")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Column Types ---")
    print(df.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Unique Values in Categorical Columns ---")
    print(f"ChannelId: {df['ChannelId'].unique()}")
    print(f"ProviderId: {df['ProviderId'].unique()}")
    print(f"ProductCategory: {df['ProductCategory'].unique()}")
    print(f"PricingStrategy: {df['PricingStrategy'].unique()}")
    print(f"CurrencyCode: {df['CurrencyCode'].unique()}")
    print(f"CountryCode: {df['CountryCode'].unique()}")
    
    print("\n--- Data Range ---")
    if 'TransactionStartTime' in df.columns:
        print(f"TransactionStartTime Min: {df['TransactionStartTime'].min()}")
        print(f"TransactionStartTime Max: {df['TransactionStartTime'].max()}")

else:
    print(f"File {data_path} not found.")
