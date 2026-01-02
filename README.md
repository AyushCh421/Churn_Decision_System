# 📉 Customer Churn Decision System

A machine learning–based decision-support system that predicts customer churn probability, assigns risk levels, and provides a business-optimized churn decision.

Built as an end-to-end applied ML project covering data cleaning, modeling, risk segmentation, threshold optimization, and deployment.

---

## 🚀 Project Overview

Customer churn is a critical problem for subscription-based businesses.  
This project aims to:

- Predict the **probability of customer churn**
- Segment customers into **Low / Medium / High risk**
- Apply a **business-optimized decision threshold** instead of a default ML cutoff
- Deploy the model as an **interactive web application**

---

## 🧠 Approach

1. **Data Cleaning & Preprocessing**
   - Removed data leakage features (e.g., churn score, CLTV)
   - Handled missing values
   - Performed encoding and feature scaling

2. **Model Training**
   - Logistic Regression used for stable and interpretable predictions
   - Evaluated using Precision, Recall, ROC-AUC

3. **Probability-Based Risk Segmentation**
   - Low Risk, Medium Risk, High Risk buckets created from churn probability

4. **Threshold Optimization**
   - Tested multiple probability thresholds
   - Final threshold selected at **0.40** to minimize missed churners
   - Decision based on business cost considerations

5. **Deployment**
   - Deployed as a Streamlit app for real-time churn prediction

---

## 📊 Key Features

- Churn probability prediction
- Risk level classification
- Business-optimized churn decision
- Clean and simple UI
- End-to-end ML pipeline

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

Customer-Churn-Decision-System/
│
├── app.py
├── logistic_churn_model.pkl
├── scaler.pkl
├── model_features.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│ └── telco.csv
│
└── notebooks/
├── 01_Data_Cleaning_and_model.ipynb
├── Probability_Segmentation.ipynb
└── Threshold_optimisation.ipynb



---

## ▶️ How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
streamlit run app.py



🌐 Deployment
The application is deployed using Streamlit Community Cloud.

(Add deployment link here after hosting)

👤 Author

Ayush Chauhan
ML Enthusiast