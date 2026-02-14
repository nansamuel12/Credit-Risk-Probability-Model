
import pandas as pd

df = pd.read_csv('data/raw/data.csv')
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
