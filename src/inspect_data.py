
import pandas as pd

try:
    df = pd.read_csv('data/raw/data.csv')
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head().to_string())
    print("\nValue Counts for FraudResult:")
    print(df['FraudResult'].value_counts())
    print("\nValue Counts for ProductCategory:")
    print(df['ProductCategory'].value_counts())
    print("\nDescriptive Stats for Amount and Value:")
    print(df[['Amount', 'Value']].describe())
except Exception as e:
    print(e)
