import streamlit as st
import numpy as np
import pickle

st.title("Coal Price Forecasting")

# Load the trained XGBoost model
with open("xgboost_coal_forecasting.pkl", "rb") as file:
    model = pickle.load(file)

# Sidebar inputs
st.sidebar.header("Input Parameters")
gdp = st.sidebar.number_input("GDP Growth (%)", min_value=0.0, step=0.1)
inflation_rate = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, step=0.1)
exchange_rate = st.sidebar.number_input("Exchange Rate (USD)", min_value=0.0, step=0.1)
nat_gas_price = st.sidebar.number_input("Natural Gas Price (USD/MMBtu)", min_value=0.0, step=0.1)
coal_production = st.sidebar.number_input("Coal Production", min_value=0.0, step=0.1)
oil_price = st.sidebar.number_input("Oil Price (USD per barrel)", min_value=0.0, step=0.1)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
unemployment_rate = st.sidebar.number_input("Unemployment Rate (%)", min_value=0.0, step=0.1)
industrial_production = st.sidebar.number_input("Industrial Production Index", min_value=0.0, step=0.1)

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[gdp, inflation_rate, exchange_rate, nat_gas_price,
                           coal_production, oil_price, interest_rate, unemployment_rate,
                           industrial_production]], dtype=np.float32)
    st.write("Input Data:", input_data)  # Debugging
    prediction = model.predict(input_data)
    st.write(f"### 🔥 Predicted Coal Price: **${prediction[0]:.2f} per ton**")
