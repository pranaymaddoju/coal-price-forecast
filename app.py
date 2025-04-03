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
oil_price = st.number_input("Oil Price Index", value=50.0)
natural_gas_price = st.number_input("Natural Gas Price", value=3.5)
gdp = st.number_input("GDP", value=20000.0)
exchange_rate = st.number_input("Exchange Rate", value=1.2)

# ✅ Make Prediction
if st.button("Predict Coal Price"):
    input_data = pd.DataFrame([[oil_price, natural_gas_price, gdp, exchange_rate]],
                              columns=["Oil_Price_Index", "Natural_Gas_Price", "GDP", "Exchange_Rate"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Coal Price: {prediction:.2f}")
