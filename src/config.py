
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class PathConfig:
    raw_data: Path = Path("data/data.csv")
    processed_data: Path = Path("data/processed/customer_risk_data.csv")
    models_dir: Path = Path("models")
    risk_model: Path = Path("models/risk_model.pkl")
    amount_model: Path = Path("models/amount_model.pkl")
    preprocessor: Path = Path("models/preprocessor.pkl")

@dataclass
class ColumnConfig:
    target: str = "Risk_Label"
    id_col: str = "CustomerId"
    date_col: str = "TransactionStartTime"
    amount_col: str = "Amount"
    value_col: str = "Value"
    fraud_label: str = "FraudResult"
    
    # Categorical columns for feature engineering
    cat_cols: List[str] = field(default_factory=lambda: [
        'ChannelId', 'ProductCategory', 'PricingStrategy', 'ProviderId'
    ])
    
    # Features expected by the model (derived)
    required_features: List[str] = field(default_factory=lambda: [
        'Total_Transactions', 'Total_Amount', 'Average_Amount', 'Amount_Std', 'Amount_min', 'Amount_max', 
        'Total_Value', 'Value_mean', 'TransactionHour_mean', 
        'ChannelId_ChannelId_1_sum', 'ChannelId_ChannelId_2_sum', 'ChannelId_ChannelId_3_sum', 'ChannelId_ChannelId_5_sum', 
        'ProductCategory_airtime_sum', 'ProductCategory_data_bundles_sum', 'ProductCategory_financial_services_sum', 
        'ProductCategory_movies_sum', 'ProductCategory_other_sum', 'ProductCategory_ticket_sum', 'ProductCategory_transport_sum', 
        'ProductCategory_tv_sum', 'ProductCategory_utility_bill_sum', 
        'PricingStrategy_0_sum', 'PricingStrategy_1_sum', 'PricingStrategy_2_sum', 'PricingStrategy_4_sum'
    ])

@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    n_clusters_risk: int = 3
    rf_n_estimators: int = 100
    rf_max_depth: int = 10

@dataclass
class ProjectConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    cols: ColumnConfig = field(default_factory=ColumnConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

# Global configuration instance
config = ProjectConfig()
