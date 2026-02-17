
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path to allow direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Ensure output directory exists
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
# Fixed data path (from data/raw/data.csv to data/data.csv)
df = pd.read_csv('data/data.csv')
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

print("Generating Distribution of Transaction Amounts...")
plt.figure(figsize=(12, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.savefig(f'{output_dir}/distribution_amount.png')
plt.close()

print("Generating Log-Distribution of Transaction Values...")
plt.figure(figsize=(12, 6))
sns.histplot(df['Value'], bins=50, kde=True, log_scale=True)
plt.title('Log-Distribution of Transaction Values')
plt.xlabel('Value (Log Scale)')
plt.savefig(f'{output_dir}/distribution_value_log.png')
plt.close()

print("Generating Fraud Class Imbalance...")
plt.figure(figsize=(6, 4))
sns.countplot(x='FraudResult', data=df)
plt.title('Fraud Class Imbalance')
plt.yscale('log')
plt.savefig(f'{output_dir}/fraud_imbalance.png')
plt.close()


print("Generating Correlation Matrix...")
numerical_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
corr = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig(f'{output_dir}/correlation_matrix.png')
plt.close()

print("Generating Boxplot of Transaction Amounts...")
plt.figure(figsize=(12, 4))
sns.boxplot(x=df['Amount'])
plt.title('Boxplot of Transaction Amounts')
plt.savefig(f'{output_dir}/boxplot_amount.png')
plt.close()

# --- New Components ---

print("Generating Dataset Overview (Summary Table)...")
# Create a figure to hold the text/table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Prepare text/table content
summary_stats = df[['Amount', 'Value', 'PricingStrategy']].describe().round(2)
# Convert DataFrame to table text for simple rendering or use table function
# Using matplotlib table
table_data = [summary_stats.columns.tolist()] + summary_stats.reset_index().values.tolist()
# Actually, reset_index makes 'index' a column.
# Let's just make a nice table of describe()
desc_df = df[['Amount', 'Value', 'PricingStrategy']].describe().round(2)
table = ax.table(cellText=desc_df.values,
                 rowLabels=desc_df.index,
                 colLabels=desc_df.columns,
                 loc='center', 
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Dataset Summary Statistics (Numerical)', fontsize=14, y=0.9)
plt.savefig(f'{output_dir}/dataset_summary_table.png', bbox_inches='tight')
plt.close()

print("Generating Categorical Distribution: ProductCategory...")
plt.figure(figsize=(12, 8))
# Order by count
order = df['ProductCategory'].value_counts().index
sns.countplot(y='ProductCategory', data=df, order=order, palette='viridis')
plt.title('Distribution of Product Categories')
plt.xlabel('Count')
plt.ylabel('Product Category')
plt.savefig(f'{output_dir}/distribution_product_category.png', bbox_inches='tight')
plt.close()


print("Generating Categorical Distribution: ChannelId...")
plt.figure(figsize=(8, 5))
order_channel = df['ChannelId'].value_counts().index
sns.countplot(x='ChannelId', data=df, order=order_channel, palette='magma')
plt.title('Distribution of Channel IDs')
plt.xlabel('Channel ID')
plt.ylabel('Count')
plt.savefig(f'{output_dir}/distribution_channel_id.png', bbox_inches='tight')
plt.close()

print("Generating Categorical Distribution: ProviderId...")
plt.figure(figsize=(10, 6))
order_provider = df['ProviderId'].value_counts().index
sns.countplot(y='ProviderId', data=df, order=order_provider, palette='cool')
plt.title('Distribution of Provider IDs')
plt.xlabel('Count')
plt.ylabel('Provider ID')
plt.savefig(f'{output_dir}/distribution_provider_id.png', bbox_inches='tight')
plt.close()

print(f"All figures saved to {output_dir}/")
