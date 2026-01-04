import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Decision System",
    page_icon="📉",
    layout="centered"
)

st.title("📊 Customer Churn Decision System")
st.caption("Built by Ayush Chauhan — ML Enthusiast")

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()
NUM_COLS = list(scaler.feature_names_in_)

# ----------------------------
# User Inputs
# ----------------------------
st.header("Enter Customer Details")

tenure = st.slider("Tenure (Months)", 0, 72, 1)
monthly_charge = st.slider("Monthly Charge", 20, 150, 90)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Type",
    ["DSL", "Fiber Optic", "No Internet"]
)

online_security = st.selectbox("Online Security", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Predict Churn Risk"):

    # Initialize full feature vector
    input_df = pd.DataFrame(0, index=[0], columns=FEATURES)

    # Numeric features
    input_df.at[0, "Tenure in Months"] = tenure
    input_df.at[0, "Monthly Charge"] = monthly_charge

    # Contract encoding
    if contract == "One year":
        input_df.at[0, "Contract_One Year"] = 1
    elif contract == "Two year":
        input_df.at[0, "Contract_Two Year"] = 1

    # Internet encoding
    if internet == "Fiber Optic":
        input_df.at[0, "Internet Type_Fiber Optic"] = 1
    elif internet == "DSL":
        input_df.at[0, "Internet Type_DSL"] = 1

    # Other services
    if online_security == "Yes":
        input_df.at[0, "Online Security_Yes"] = 1

    if paperless == "Yes":
        input_df.at[0, "Paperless Billing_Yes"] = 1

    # Scale numeric columns ONLY
    input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    # Output
    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")

    if prob >= 0.6:
        st.error("⚠️ High Risk — Likely to Churn")
    else:
        st.success("✅ Low Risk — Not Likely to Churn")
