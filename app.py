import numpy as np
import pandas as pd
import streamlit as st
import pickle

# ✅ Load the trained XGBoost model
MODEL_PATH = "xgboost_coal_forecasting (3).pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# ✅ Streamlit App
st.title("Coal Price Forecasting App")

# ✅ User Input for External Factors
oil_price_index = st.number_input("Oil Price Index", value=0.0)
natural_gas_price = st.number_input("Natural Gas Price", value=0.0)
gdp = st.number_input("GDP", value=0.0)
exchange_rate = st.number_input("Exchange Rate", value=0.0)
oil_price_lag7 = st.number_input("Oil Price Lag 7", value=0.0)
gas_price_lag7 = st.number_input("Gas Price Lag 7", value=0.0)
exchange_rate_lag7 = st.number_input("Exchange Rate Lag 7", value=0.0)
oil_price_rolling14 = st.number_input("Oil Price Rolling 14", value=0.0)
gas_price_rolling14 = st.number_input("Gas Price Rolling 14", value=0.0)

# ✅ Make Prediction
if st.button("Predict Coal Price"):
    input_data = np.array([[
        oil_price_index, natural_gas_price, gdp, exchange_rate,
        oil_price_lag7, gas_price_lag7, exchange_rate_lag7,
        oil_price_rolling14, gas_price_rolling14
    ]])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Coal Price: {prediction:.2f}")
