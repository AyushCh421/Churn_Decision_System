import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load model artifacts
# ===============================
model = joblib.load("logistic_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

FINAL_THRESHOLD = 0.40

# ===============================
# App UI
# ===============================
st.set_page_config(page_title="Customer Churn Decision System", layout="centered")

st.title("📉 Customer Churn Decision System")

st.caption("Built by Ayush Chauhan — ML Enthusiast")


st.write(
    "This app predicts **churn probability**, assigns a **risk level**, "
    "and provides a **business-optimized churn decision**."
)

st.markdown("---")

# ===============================
# User Inputs
# ===============================
st.subheader("Enter Customer Details")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charge = st.slider("Monthly Charge", 20, 150, 70)
total_charges = st.slider("Total Charges", 0, 10000, 2000)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_type = st.selectbox("Internet Type", ["DSL", "Fiber optic", "No Internet"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.markdown("---")

# ===============================
# Prepare input data
# ===============================
input_df = pd.DataFrame(columns=model_features)
input_df.loc[0] = 0  # initialize all features with 0

# Numerical features
input_df["Tenure in Months"] = tenure
input_df["Monthly Charge"] = monthly_charge
input_df["Total Charges"] = total_charges

# Categorical encodings (match training)
if contract == "Month-to-month":
    col = "Contract_Month-to-month"
elif contract == "One year":
    col = "Contract_One year"
else:
    col = "Contract_Two year"
if col in input_df.columns:
    input_df[col] = 1

if internet_type != "No Internet":
    col = f"Internet Type_{internet_type}"
    if col in input_df.columns:
        input_df[col] = 1

if online_security == "Yes" and "Online Security_Yes" in input_df.columns:
    input_df["Online Security_Yes"] = 1

if paperless_billing == "Yes" and "Paperless Billing_Yes" in input_df.columns:
    input_df["Paperless Billing_Yes"] = 1

# ===============================
# Scale numerical features
# ===============================
num_cols = scaler.feature_names_in_
input_df[num_cols] = scaler.transform(input_df[num_cols])

# ===============================
# Prediction
# ===============================
if st.button("Predict Churn Risk"):
    churn_prob = model.predict_proba(input_df)[0][1]

    # Risk segmentation
    if churn_prob >= 0.60:
        risk_level = "🔴 High Risk"
    elif churn_prob >= 0.30:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🟢 Low Risk"

    decision = (
        "❗ Likely to Churn"
        if churn_prob >= FINAL_THRESHOLD
        else "✅ Not Likely to Churn"
    )

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {churn_prob:.2%}")
    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Final Decision:** {decision}")

    st.markdown("---")
    st.caption(
        "Decision threshold optimized at 0.40 to minimize missed churners "
        "based on business cost considerations."
    )
