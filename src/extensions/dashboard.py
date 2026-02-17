
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import config

# Streamlit Config
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

@st.cache_resource
def load_models():
    risk_model = joblib.load(config.paths.risk_model)
    amount_model = joblib.load(config.paths.amount_model)
    preprocessor = joblib.load(config.paths.preprocessor)
    return risk_model, amount_model, preprocessor

try:
    risk_model, amount_model, preprocessor = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("Credit Risk Assessment Dashboard")

# Sidebar for Input
st.sidebar.header("User Features")

def user_input_features():
    total_trx = st.sidebar.number_input("Total Transactions", min_value=0, value=5)
    total_amt = st.sidebar.number_input("Total Amount", min_value=0.0, value=5000.0)
    avg_amt = st.sidebar.number_input("Average Amount", min_value=0.0, value=1000.0)
    amt_std = st.sidebar.number_input("Amount Std Dev", min_value=0.0, value=100.0)
    amt_min = st.sidebar.number_input("Min Amount", min_value=0.0, value=500.0)
    amt_max = st.sidebar.number_input("Max Amount", min_value=0.0, value=2000.0)
    total_val = st.sidebar.number_input("Total Value", min_value=0.0, value=5000.0)
    val_mean = st.sidebar.number_input("Value Mean", min_value=0.0, value=1000.0)
    
    features = {f: 0.0 for f in config.cols.required_features}
    features.update({
        'Total_Transactions': total_trx,
        'Total_Amount': total_amt,
        'Average_Amount': avg_amt,
        'Amount_Std': amt_std,
        'Amount_min': amt_min,
        'Amount_max': amt_max,
        'Total_Value': total_val,
        'Value_mean': val_mean,
        config.cols.id_col: 0,
        'Cluster': 0,
        'FraudResult_max': 0,
        'Risk_Label': 0
    })
    return pd.DataFrame([features])

input_df = user_input_features()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if st.button("Assess Risk"):
        try:
            X_scaled = preprocessor.transform(input_df)
            if hasattr(risk_model, 'n_features_in_') and X_scaled.shape[1] > risk_model.n_features_in_:
                X_scaled = X_scaled[:, :risk_model.n_features_in_]
            
            risk_prob = risk_model.predict_proba(X_scaled)[0][1]
            credit_score = int(300 + (1 - risk_prob) * 550)
            risk_cat = "High Risk" if risk_prob > 0.5 else "Low Risk"
            
            st.metric("Risk Probability", f"{risk_prob:.2%}")
            st.metric("Credit Score", credit_score)
            st.metric("Risk Category", risk_cat, delta_color="inverse")
            
            st.subheader("Decision Breakdown (SHAP)")
            explainer = shap.TreeExplainer(risk_model)
            shap_values = explainer.shap_values(X_scaled)
            
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_scaled[0,:]))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

with col2:
    st.subheader("Historical Data Inspector")
    if os.path.exists(config.paths.processed_data):
        df_hist = pd.read_csv(config.paths.processed_data).head(100)
        st.dataframe(df_hist)
        
        st.subheader("Feature Distributions")
        feat = st.selectbox("Select Feature", config.cols.required_features[:5])
        st.bar_chart(df_hist[feat])
