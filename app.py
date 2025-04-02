import os
import streamlit as st
import numpy as np
import pickle

st.title("Coal Price Forecasting")

# ✅ Load the model
model_path = os.path.join(os.path.dirname(__file__), "xgboost_coal_forecasting.pkl")

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'xgboost_coal_forecasting.pkl' is in your GitHub repository.")
    st.stop()

# Input features
st.sidebar.header("Input Parameters")
gdp = st.sidebar.number_input("GDP Growth (%)", min_value=0.0, step=0.1)
inflation = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, step=0.1)
exchange_rate = st.sidebar.number_input("Exchange Rate (USD)", min_value=0.0, step=0.1)
nat_gas_price = st.sidebar.number_input("Natural Gas Price (USD/MMBtu)", min_value=0.0, step=0.1)

# Predict
if st.sidebar.button("Predict"):
    if model:
        # ✅ Convert input to a NumPy array and reshape it
        input_data = np.array([[gdp, inflation, exchange_rate, nat_gas_price]], dtype=np.float32)
        
        # ✅ Convert to DMatrix if model expects it
        try:
            input_dmatrix = xgb.DMatrix(input_data)
            prediction = model.predict(input_dmatrix)
        except:
            prediction = model.predict(input_data)

        st.write(f"### Predicted Coal Price: ${prediction[0]:.2f} per ton")
    else:
        st.error("❌ Model is not loaded. Check if 'xgboost_coal_forecasting.pkl' exists in your GitHub repo.")
