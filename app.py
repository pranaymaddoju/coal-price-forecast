import streamlit as st
import pickle
import numpy as np

# Set Streamlit app title
st.title("Coal Price Forecasting")

# Load the trained XGBoost model
MODEL_PATH = "xgboost_coal_forecasting (2).pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model Loaded Successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_PATH}' not found! Please upload the correct model file.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Input Features")
coal_production = st.sidebar.number_input("Coal Production", min_value=0.0, format="%.2f")
oil_price = st.sidebar.number_input("Oil Price", min_value=0.0, format="%.2f")
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, format="%.2f")
unemployment_rate = st.sidebar.number_input("Unemployment Rate", min_value=0.0, format="%.2f")
industrial_production = st.sidebar.number_input("Industrial Production", min_value=0.0, format="%.2f")

# Predict button
if st.sidebar.button("Predict Coal Price"):
    input_data = np.array([[coal_production, oil_price, interest_rate, unemployment_rate, industrial_production]], dtype=np.float32)
    st.write("### 🔍 Input Data:", input_data)  # Debugging
    
    try:
        prediction = model.predict(input_data)
        st.write(f"### 🔥 Predicted Coal Price: **${prediction[0]:.2f} per ton**")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
