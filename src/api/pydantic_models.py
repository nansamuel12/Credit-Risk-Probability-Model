

from pydantic import BaseModel
from typing import Optional

class CustomerFeatures(BaseModel):
    CustomerId: str
    Total_Transactions: int
    Total_Amount: float
    Average_Amount: float
    Amount_std: float = 0.0
    Amount_min: float
    Amount_max: float
    Total_Value: float
    Value_mean: float

class PredictionResponse(BaseModel):
    CustomerId: str
    RiskProbability: float
    CreditScore: int
    OptimalLoanAmount: float
    RiskCategory: str
