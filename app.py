import os
import streamlit as st
import numpy as np
import pickle
import xgboost as xgb

st.title("Coal Price Forecasting")

# ✅ Load the model
model_filename = "xgboost_coal_forecasting.pkl"  # Ensure this matches your file
model_path = os.path.join(os.path.dirname(__file__), model_filename)

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file '{model_filename}' not found. Make sure it is in your GitHub repository.")
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
        # ✅ Ensure input is a 2D NumPy array
        input_data = np.array([[gdp, inflation, exchange_rate, nat_gas_price]], dtype=np.float32)

        # ✅ Convert input to match model's training format
        if isinstance(model, xgb.XGBRegressor):  # Check if the model is an XGBoost regressor
            prediction = model.predict(input_data)
        else:
            input_dmatrix = xgb.DMatrix(input_data)
            prediction = model.predict(input_dmatrix)

        st.write(f"### Predicted Coal Price: ${prediction[0]:.2f} per ton")
    else:
        st.error("❌ Model is not loaded. Check if the model file exists in your GitHub repo.")
