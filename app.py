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
inflation = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, step=0.1)
exchange_rate = st.sidebar.number_input("Exchange Rate (USD)", min_value=0.0, step=0.1)
nat_gas_price = st.sidebar.number_input("Natural Gas Price (USD/MMBtu)", min_value=0.0, step=0.1)

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[gdp, inflation, exchange_rate, nat_gas_price]], dtype=np.float32)
    
    # ✅ Directly predict using the model (No need for DMatrix)
    prediction = model.predict(input_data)

    st.write(f"### 🔥 Predicted Coal Price: **${prediction[0]:.2f} per ton**")
