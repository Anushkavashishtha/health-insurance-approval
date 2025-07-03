# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Claim Approval Predictor", layout="centered")
st.title("Insurance Claim Approval Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()

# User Inputs
st.subheader("Enter Claim Details")
age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)
claim_amount = st.number_input("Claim Amount", min_value=100.0, step=100.0)
accident_type = st.selectbox("Accident Type", ["home", "car", "work", "natural disaster"])

# Preprocessing
accident_type_dict = {"home": 0, "car": 1, "work": 2, "natural disaster": 3}
input_df = pd.DataFrame([{
    "age": age,
    "claim_amount": claim_amount,
    "accident_type": accident_type_dict[accident_type]
}])

# Predict
if st.button("Predict Approval"):
    proba = model.predict_proba(input_df)[0][1]  # Probability of approval
    prediction = model.predict(input_df)[0]

    st.markdown(f"### 🔍 Approval Probability: **{proba * 100:.2f}%**")

    if prediction == 1:
        st.success(" Claim Approved")
    else:
        st.error(" Claim Rejected")

