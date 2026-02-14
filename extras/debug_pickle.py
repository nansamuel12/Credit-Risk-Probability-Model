
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

try:
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("Preprocessor loaded successfully.")
    
    # Create dummy data
    # Based on main.py REQUIRED_FEATURES
    # Wait, the preprocessor expects columns from process_data (e.g. Frequency, Recency)??
    # OR columns from main.py?
    # The crash happened inside main.py with 'CustomerId', 'Risk_Label' missing.
    # So we know it checks columns.
    
    # Let's try to mimic main.py's input exactly
    input_data = {
        'Total_Transactions': 5, 
        'Total_Amount': 500.0, 
        'Average_Amount': 100.0, 
        'Amount_Std': 10.0, 
        'Amount_min': 90.0, 
        'Amount_max': 110.0, 
        'Total_Value': 600.0, 
        'Value_mean': 120.0,
        # ... other fields from main.py REQUIRED_FEATURES initialized to 0
    }
    REQUIRED_FEATURES = [
        'Total_Transactions', 'Total_Amount', 'Average_Amount', 'Amount_Std', 'Amount_min', 'Amount_max', 
        'Total_Value', 'Value_mean', 'TransactionHour_mean', 
        'ChannelId_ChannelId_1_sum', 'ChannelId_ChannelId_2_sum', 'ChannelId_ChannelId_3_sum', 'ChannelId_ChannelId_5_sum', 
        'ProductCategory_airtime_sum', 'ProductCategory_data_bundles_sum', 'ProductCategory_financial_services_sum', 
        'ProductCategory_movies_sum', 'ProductCategory_other_sum', 'ProductCategory_ticket_sum', 'ProductCategory_transport_sum', 
        'ProductCategory_tv_sum', 'ProductCategory_utility_bill_sum', 
        'PricingStrategy_0_sum', 'PricingStrategy_1_sum', 'PricingStrategy_2_sum', 'PricingStrategy_4_sum'
    ]
    for f in REQUIRED_FEATURES:
        if f not in input_data:
            input_data[f] = 0.0
            
    # Add the fix columns
    input_data['CustomerId'] = 'C1'
    input_data['Risk_Label'] = 0
    input_data['Cluster'] = 0
    input_data['FraudResult_max'] = 0
    
    df = pd.DataFrame([input_data])
    
    print("Attempting transform...")
    try:
        preprocessor.transform(df)
        print("Transform success!")
    except Exception as e:
        print(f"Transform failed: {e}")
        
        # Access the imputer
        # Structure from build_training_pipeline:
        # Check preprocessor.transformers_
        # It's a ColumnTransformer.
        # transformers_ is a list of (name, transformer, columns)
        
        print("\nInspecting transformers...")
        try:
            # We iterate and patch
            for name, trans, cols in preprocessor.transformers_:
                if hasattr(trans, 'steps'): # Pipeline
                    for step_name, step in trans.steps:
                        if step_name == 'imputer':
                            print(f"Found imputer in {name}: {step}")
                            # Patch
                            if not hasattr(step, '_fill_dtype'):
                                print(f"Patching _fill_dtype for {step}")
                                step._fill_dtype = np.float64 # or object?
                            else:
                                print(f"Imputer has _fill_dtype: {step._fill_dtype}")
        except Exception as inspect_e:
            print(f"Inspection failed: {inspect_e}")

        # Retry
        print("\nRetrying transform after patch...")
        try:
            preprocessor.transform(df)
            print("Transform success after patch!")
        except Exception as e2:
            print(f"Transform failed again: {e2}")

except Exception as e:
    print(f"Global error: {e}")
