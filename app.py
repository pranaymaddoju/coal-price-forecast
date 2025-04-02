import os
import streamlit as st
import numpy as np
import joblib  # ✅ Use joblib for XGBoost models
import xgboost as xgb

st.title("Coal Price Forecasting")

# ✅ Load Model
model_filename = "xgboost_coal_forecasting.pkl"
model_path = os.path.join(os.path.dirname(__file__), model_filename)

try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file '{model_filename}' not found. Check your GitHub repository.")
    st.stop()

# 📌 Sidebar Input Features
st.sidebar.header("Input Parameters")
gdp = st.sidebar.number_input("GDP Growth (%)", min_value=0.0, step=0.1)
inflation = st.sidebar.number_input("Inflation Rate (%)", min_value=0.0, step=0.1)
exchange_rate = st.sidebar.number_input("Exchange Rate (USD)", min_value=0.0, step=0.1)
nat_gas_price = st.sidebar.number_input("Natural Gas Price (USD/MMBtu)", min_value=0.0, step=0.1)

# ✅ Prediction
if st.sidebar.button("Predict"):
    if model:
        # ✅ Ensure correct input format
        input_data = np.array([[gdp, inflation, exchange_rate, nat_gas_price]], dtype=np.float32)

        # Debugging Information
        st.write(f"🟢 **Shape of input_data:** {input_data.shape}")
        st.write(f"🟢 **Data type of input_data:** {input_data.dtype}")

        try:
            # ✅ Convert input to DMatrix before prediction
            input_dmatrix = xgb.DMatrix(input_data)
            prediction = model.predict(input_dmatrix)

            st.write(f"### Predicted Coal Price: **${prediction[0]:.2f} per ton**")
        except ValueError as e:
            st.error(f"🚨 Error during prediction: {e}")
    else:
        st.error("❌ Model is not loaded. Ensure the model file is in the GitHub repository.")
