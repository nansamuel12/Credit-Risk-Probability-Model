

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import os

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts date features from TransactionStartTime"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(X['TransactionStartTime']):
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregates transaction data to customer level."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # We need to handle OneHotEncoding manually or via get_dummies here to aggregate cleanly
        # Because standard OHE in Pipeline returns a sparse matrix which is hard to groupby
        
        # 1. Date Features and Basics are already in X from previous step
        
        # 2. One Hot Encode Categoricals for Count aggregation
        # Categories: 'ChannelId', 'ProductCategory', 'ProviderId', 'PricingStrategy'
        # We use pd.get_dummies for simplicity in this custom aggregation block
        cat_cols = ['ChannelId', 'ProductCategory', 'PricingStrategy']
        # Check if columns exist
        existing_cats = [c for c in cat_cols if c in X.columns]
        X_encoded = pd.get_dummies(X, columns=existing_cats, prefix=existing_cats)
        
        # 3. Define Aggregations
        # Numerical stats
        agg_rules = {
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Value': ['sum', 'mean'],
            'TransactionHour': 'mean', # Average time of day they transact
            'FraudResult': 'max' # Risk Label
        }
        
        # Add the one-hot columns to aggregation (summing them = count)
        for col in X_encoded.columns:
            for cat in existing_cats:
                if col.startswith(f"{cat}_"):
                    agg_rules[col] = 'sum'

        # Groupby
        customer_stats = X_encoded.groupby('CustomerId').agg(agg_rules)
        
        # Flatten MultiIndex
        customer_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in customer_stats.columns.values]
        
        # Rename explicit columns

        rename_map = {
            'TransactionId_count': 'Total_Transactions',
            'Amount_sum': 'Total_Amount',
            'Amount_mean': 'Average_Amount',
            'Amount_std': 'Amount_Std',
            'Value_sum': 'Total_Value',
            'FraudResult_max': 'Risk_Label'
        }
        # Handle dynamic renaming for the flattened columns
        # (The list comp above handles the _join, so 'TransactionId_count' exists)
        
        customer_stats.rename(columns=rename_map, inplace=True)
        
        # Reset index to make CustomerId a column
        customer_stats.reset_index(inplace=True)
        
        return customer_stats

# Pipeline Construction
def build_processing_pipeline(numerical_cols):
    """
    Builds the final processing pipeline.
    Note: The Aggregation step changes the structure significantly, 
    so typically we run Aggregation *then* this pipeline.
    However, the user asked to chain everything.
    We will create a wrapper function that runs the whole flow.
    """
    
    # Post-Aggregation Steps: Impute -> Scale
    # We only apply this to the computed numerical features, not the ID or Label
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Handle NaN std dev for single transactions
        ('scaler', StandardScaler()) # Standardize
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols)
        ],
        remainder='passthrough' # Keep ID and Label
    )
    
    return preprocessor

def run_pipeline(input_path, output_path):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Step 1: Feature Extraction (Date)
    print("Extracting Date Features...")
    date_extractor = DateFeatureExtractor()
    df_dates = date_extractor.transform(df)
    
    # Step 2: Aggregation
    print("Aggregating to Customer Level...")
    aggregator = CustomerAggregator()
    df_agg = aggregator.transform(df_dates)
    
    # Identify Numerical Columns for scaling (exclude ID and Label)
    exclude_cols = ['CustomerId', 'Risk_Label']
    numerical_cols = [c for c in df_agg.columns if c not in exclude_cols]
    
    # Step 3: Final Pipeline (Impute & Scale)
    print("Running Transformation Pipeline (Impute, Scale)...")
    pipeline = build_processing_pipeline(numerical_cols)
    
    # Fit Transform
    # ColumnTransformer returns an array, we need to reconstruct DataFrame to keep column names
    processed_array = pipeline.fit_transform(df_agg)
    
    # Get feature names from ColumnTransformer
    # 'remainder' columns (passed through) are at the end
    # The order is: [Transformed Numerics] + [Passthrough Columns]
    
    new_numeric_cols = numerical_cols # Transformer keeps order
    passthrough_cols = exclude_cols # Remainder
    all_cols = new_numeric_cols + passthrough_cols
    
    df_final = pd.DataFrame(processed_array, columns=all_cols)
    
    # Fix Types (Label should be int)
    df_final['Risk_Label'] = df_final['Risk_Label'].astype(int)
    

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    print(df_final.head())
    
    # Save Preprocessor for Inference
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/preprocessor.pkl')
    print("Saved preprocessor to models/preprocessor.pkl")
    
    return df_final

if __name__ == "__main__":
    run_pipeline('data/raw/data.csv', 'data/processed/customer_risk_data_pipeline.csv')
