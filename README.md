

# Credit Risk Probability Model

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
- **Selected Approach**: We use **Random Forest** for its high performance on complex, non-linear transaction data, but we mitigate opacity by analyzing **Feature Importance** (e.g., heavily weighting `Total_Amount` and `Frequency`) to explain decisions.

## Structure
- `data/`: Raw and processed data
- `notebooks/`: EDA and analysis
- `src/`: Source code for processing, training, and API
- `tests/`: Unit tests
