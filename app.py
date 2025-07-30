import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("💼 Customer Churn Prediction App")
st.markdown("Use this app to predict whether a bank customer will **churn** or not based on their details.")

# Create layout columns with spacing
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    st.markdown("<br>", unsafe_allow_html=True)

    gender = st.selectbox("🧑 Gender", label_encoder_gender.classes_)
    st.markdown("<br>", unsafe_allow_html=True)

    age = st.slider("🎂 Age", 18, 95)
    st.markdown("<br>", unsafe_allow_html=True)

    credit_score = st.number_input("💳 Credit Score", min_value=0)
    st.markdown("<br>", unsafe_allow_html=True)

    balance = st.number_input("💰 Account Balance", min_value=0.0)

with col2:
    tenure = st.slider("📆 Tenure (Years)", 0, 10)
    st.markdown("<br>", unsafe_allow_html=True)

    num_of_products = st.slider("📦 Number of Products", 1, 4)
    st.markdown("<br>", unsafe_allow_html=True)

    has_cr_card = st.selectbox("💳 Has Credit Card?", ["Yes", "No"])
    st.markdown("<br>", unsafe_allow_html=True)

    is_active_member = st.selectbox("✅ Is Active Member?", ["Yes", "No"])
    st.markdown("<br>", unsafe_allow_html=True)

    estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0)

# Convert Yes/No to 1/0
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.subheader("🔍 Prediction Result:")
st.write(f"**Churn Probability:** `{prediction_prob:.2f}`")

if prediction_prob > 0.5:
    st.warning("⚠️ The customer is **likely to churn.**")
else:
    st.success("✅ The customer is **not likely to churn.**")