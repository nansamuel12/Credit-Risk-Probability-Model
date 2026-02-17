import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Fixed data path
try:
    df = pd.read_csv('data/data.csv')
except FileNotFoundError:
    # Try alternate location if run from different CWD
    df = pd.read_csv('../../data/data.csv')

fin_serv = df[df['ProductCategory'] == 'financial_services']

print("Financial Services Amount Description:")
print(fin_serv['Amount'].describe())

print("\nNegative Amounts (Loans?):")
loans = fin_serv[fin_serv['Amount'] < 0]
print(loans.head())
print(f"Count of negative amounts: {len(loans)}")

print("\nPositive Amounts (Repayments?):")
repayments = fin_serv[fin_serv['Amount'] > 0]
print(repayments.head())
print(f"Count of positive amounts: {len(repayments)}")
