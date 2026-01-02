import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Decision System",
    layout="centered"
)

st.title("📊 Customer Churn Decision System")
st.caption("Built by Ayush Chauhan | ML Enthusiast")

# -----------------------------
# Load Artifacts
# -----------------------------
model = joblib.load("logistic_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("model_features.pkl")

# Numeric columns used during scaling
num_cols = list(scaler.feature_names_in_)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Customer Details")

tenure = st.slider("Tenure (Months)", 0, 72, 1)
monthly_charge = st.slider("Monthly Charge", 20, 150, 120)

contract_type = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_type = st.selectbox(
    "Internet Type",
    ["DSL", "Fiber optic", "No internet"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_df = pd.DataFrame(
    np.zeros((1, len(features))),
    columns=features
)

# Numeric features
input_df.at[0, "Tenure in Months"] = tenure
input_df.at[0, "Monthly Charge"] = monthly_charge

# Contract
if contract_type == "One year":
    input_df.at[0, "Contract_One Year"] = 1
elif contract_type == "Two year":
    input_df.at[0, "Contract_Two Year"] = 1

# Internet Type
if internet_type == "DSL":
    input_df.at[0, "Internet Type_DSL"] = 1
elif internet_type == "Fiber optic":
    input_df.at[0, "Internet Type_Fiber Optic"] = 1
else:
    input_df.at[0, "Internet Type_No Internet"] = 1

# Online Security
if online_security == "Yes":
    input_df.at[0, "Online Security_Yes"] = 1

# Paperless Billing
if paperless_billing == "Yes":
    input_df.at[0, "Paperless Billing_Yes"] = 1

# -----------------------------
# Scale Numeric Columns
# -----------------------------
input_df[num_cols] = scaler.transform(input_df[num_cols])

# -----------------------------
# Prediction
# -----------------------------
churn_prob = model.predict_proba(input_df)[0][1]

st.markdown("---")
st.subheader("Prediction Result")

st.metric(
    label="Churn Probability",
    value=f"{churn_prob * 100:.2f}%"
)

# Risk Label
if churn_prob >= 0.6:
    st.error("🔴 High Risk of Churn")
elif churn_prob >= 0.3:
    st.warning("🟡 Medium Risk of Churn")
else:
    st.success("🟢 Low Risk of Churn")

st.markdown("**Final Decision:**")
if churn_prob >= 0.5:
    st.write("❌ Likely to Churn")
else:
    st.write("✅ Not Likely to Churn")
