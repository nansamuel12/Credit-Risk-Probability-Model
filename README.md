# Credit Risk Probability Model

[![Credit Risk Model CI](https://github.com/nansamuel12/Credit-Risk-Probability-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/nansamuel12/Credit-Risk-Probability-Model/actions/workflows/ci.yml)

This project aims to build a credit risk model to categorize users, predict risk probability, assign credit scores, and predict optimal loan amounts.

## Credit Scoring Business Understanding

### Basel II Context
In the context of Basel II regulations, financial institutions are required to maintain a minimum amount of capital to cover operational and credit risks. This project implements an **Internal Ratings-Based (IRB)** approach, where the institution estimates key risk parameters:
- **PD (Probability of Default)**: The likelihood that a borrower will default (predicted by our `Risk Model`).
- **LGD (Loss Given Default)**: The amount lost if a default occurs (assumed constant or modeled separately).
- **EAD (Exposure at Default)**: The total value at risk (related to our `Optimal Loan Amount` model).

### Proxy Label Rationale
Defining "Default" in transaction data is challenging without explicit repayment labels. We use **RFMS (Recency, Frequency, Monetary, Stability)** logic to define a proxy:
- **Good (Low Risk)**: Consistent transaction history, high frequency, stable (low variance) usage.
- **Bad (High Risk)**: History of fraudulent activity (`FraudResult=1`) or highly erratic, suspicious patterns.
*Risk*: Using Fraud as a proxy for Credit Risk is imperfect. A user might be honest but unable to pay (Credit Risk), vs. acting maliciously (Fraud Risk). This model primarily assesses "integrity risk" as a baseline for creditworthiness.

### Model Interpretability vs. Complexity
- **Interpretable Models (Logistic Regression, Decision Trees)**: Preferred by regulators for transparency. Easier to explain *why* a customer was rejected.
- **Complex Models (Random Forest, Gradient Boosting)**: Often yield higher accuracy but are "black boxes".
- **Selected Approach**: We use **Random Forest** for its high performance on complex, non-linear transaction data, but we mitigate opacity by analyzing **Feature Importance** (e.g., heavily weighting `Total_Amount` and `Frequency`) to explain decisions. We also benchmark against **Logistic Regression** for interpretability.

### RFM Feature Engineering
We use **RFM (Recency, Frequency, Monetary)** features to summarize customer behavior. This transforms transaction-level data (many rows per customer) into customer-level data (one row per customer):

- **Recency**: Days since the last transaction. High recency indicates inactivity (potential risk).
- **Frequency**: Total number of transactions. Low frequency indicates low engagement or new users.
- **Monetary**: Total amount spent. Low spending indicates low customer value.

Calculation for each customer:
- `Recency = Today - Last Transaction Date`
- `Frequency = Count(Transactions)`
- `Monetary = Sum(Amount)`

## Structure
```
credit-risk-model/
│
├── data/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── rfm.py
│   ├── train.py
│
├── tests/
│
├── requirements.txt
├── README.md
```

## Deployment & MLflow

### Environment Variables
The application calculates predictions using models that can be served from a local file or an MLflow Model Registry.

- `MLFLOW_TRACKING_URI`: (Optional) The URI of your MLflow tracking server (e.g., `http://localhost:5000` or Databricks URI).
- `MLFLOW_MODEL_NAME`: Name of the registered model to load (Default: `CreditRiskModel_RF`).
- `MLFLOW_MODEL_STAGE`: Stage of the model to load (e.g., `Production`, `Staging`, `None`. Default: `Production`).

### Local Setup
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Process Data**: `python src/data_processing.py`
3. **Train Models**: `python src/train.py` (This saves to `models/` and logs to local MLflow `mlruns/`)
4. **Predict**: `python src/predict.py`

### CI/CD
This repo uses GitHub Actions to run tests (`pytest`) and linting (`flake8`) on every push to `main`.
