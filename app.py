import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd

# Set page title
st.title("🔥 Coal Price Forecasting App")

# Load the trained XGBoost model
try:
    model = xgb.Booster()
    model.load_model("xgboost_coal_forecasting.json")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load the model. Make sure 'xgboost_coal_forecasting.json' is in the directory.")
    st.stop()

# Sidebar inputs
st.sidebar.header("📊 Input Parameters")

coal_production = st.sidebar.number_input("Coal Production (tons)", min_value=0.0, value=1000.0)
oil_price = st.sidebar.number_input("Oil Price (USD per barrel)", min_value=0.0, value=70.0)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=5.0)
exchange_rate = st.sidebar.number_input("Exchange Rate (USD to INR)", min_value=50.0, value=75.0)
inflation_rate = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, value=3.0)
industrial_production = st.sidebar.number_input("Industrial Production Index", min_value=50.0, value=100.0)

# Create input data as a DataFrame
input_data = pd.DataFrame({
    "coal_production": [coal_production],
    "oil_price": [oil_price],
    "interest_rate": [interest_rate],
    "exchange_rate": [exchange_rate],
    "inflation_rate": [inflation_rate],
    "industrial_production": [industrial_production],
})

st.write("### 📋 Model Input Data")
st.dataframe(input_data)

# Make a prediction
if st.button("📈 Predict Coal Price"):
    try:
        dmatrix = xgb.DMatrix(input_data)  # Convert input data to DMatrix format
        prediction = model.predict(dmatrix)[0]
        st.success(f"🔥 Predicted Coal Price: **${prediction:.2f} per ton**")
    except Exception as e:
        st.error("❌ Prediction failed! Check the input values and model integrity.")
