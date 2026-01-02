import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Churn Decision System",
    layout="centered"
)

st.title("📉 Customer Churn Decision System")
st.caption("Built by Ayush Chauhan | ML Enthusiast")

# -------------------- LOAD ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# -------------------- USER INPUT --------------------
st.subheader("🧾 Customer Information")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20, 200, 70)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["Fiber Optic", "DSL", "No Internet"])
payment = st.selectbox(
    "Payment Method",
    ["Credit Card", "Electronic Check", "Mailed Check"]
)

# -------------------- BUILD INPUT ROW --------------------
input_dict = dict.fromkeys(FEATURES, 0)

# Numerical
input_dict["tenure"] = tenure
input_dict["MonthlyCharges"] = monthly_charges

# Contract
if contract == "One year":
    input_dict["Contract_One Year"] = 1
elif contract == "Two year":
    input_dict["Contract_Two Year"] = 1

# Internet
if internet == "Fiber Optic":
    input_dict["InternetService_Fiber optic"] = 1
elif internet == "DSL":
    input_dict["InternetService_DSL"] = 1

# Billing
if paperless == "Yes":
    input_dict["PaperlessBilling_Yes"] = 1

# Payment
if payment == "Credit Card":
    input_dict["PaymentMethod_Credit card (automatic)"] = 1
elif payment == "Electronic Check":
    input_dict["PaymentMethod_Electronic check"] = 1
elif payment == "Mailed Check":
    input_dict["PaymentMethod_Mailed check"] = 1

# -------------------- DATAFRAME --------------------
input_df = pd.DataFrame([input_dict])

# Scale numerical columns
num_cols = ["tenure", "MonthlyCharges"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Churn Risk"):
    churn_prob = model.predict_proba(input_df)[0][1]

    st.metric("Churn Probability", f"{churn_prob:.2%}")

    if churn_prob >= 0.60:
        st.error("🚨 High Risk of Churn")
    elif churn_prob >= 0.40:
        st.warning("⚠️ Medium Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.caption("Prediction based on Logistic Regression probability score")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("This model demonstrates end-to-end ML deployment: preprocessing → modeling → inference → UI.")
