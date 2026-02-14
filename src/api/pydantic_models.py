
from pydantic import BaseModel, Field
from typing import Optional

class CustomerFeatures(BaseModel):
    CustomerId: str = Field(..., description="Unique identifier for the customer")
    Total_Transactions: int = Field(..., description="Total number of transactions made by the customer")
    Total_Amount: float = Field(..., description="Sum of all transaction amounts")
    Average_Amount: float = Field(..., description="Average transaction amount")
    Amount_std: float = Field(0.0, description="Standard deviation of transaction amounts")
    Amount_min: float = Field(..., description="Minimum transaction amount")
    Amount_max: float = Field(..., description="Maximum transaction amount")
    Total_Value: float = Field(..., description="Total absolute value of transactions")
    Value_mean: float = Field(..., description="Mean absolute value of transactions")

class PredictionResponse(BaseModel):
    CustomerId: str
    RiskProbability: float
    CreditScore: int
    OptimalLoanAmount: float
    RiskCategory: str
