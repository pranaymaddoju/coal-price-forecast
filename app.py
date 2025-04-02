import os
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

st.title("Coal Price Forecasting")

# ✅ Get the correct file path
model_path = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")

# ✅ Load the model safely
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'xgboost_model.pkl' is in your GitHub repository.")

# Input features
st.sidebar.header("Input Parameters")
gdp = st.sidebar.number_input("GDP Growth (%)", min_value=0.0, step=0.1)
inflation = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, step=0.1)
exchange_rate = st.sidebar.number_input("Exchange Rate (USD)", min_value=0.0, step=0.1)
nat_gas_price = st.sidebar.number_input("Natural Gas Price (USD/MMBtu)", min_value=0.0, step=0.1)

# Predict
if st.sidebar.button("Predict"):
    if 'model' in locals():
        input_data = np.array([[gdp, inflation, exchange_rate, nat_gas_price]])
        prediction = model.predict(input_data)
        st.write(f"### Predicted Coal Price: ${prediction[0]:.2f} per ton")
    else:
        st.error("❌ Model is not loaded. Check if 'xgboost_model.pkl' exists in your GitHub repo.")

import os
st.write("Current Directory:", os.getcwd())
st.write("Files:", os.listdir(os.getcwd()))

