import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Download model from Google Drive if not already present
model_url = "https://drive.google.com/uc?id=1Zx_Wbsxgd73U93OOvJ76bT7qOTe7_Gw8"
model_path = "random_forest_model.pkl"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

# Load model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

# App config
st.set_page_config(page_title="🔋 Electricity Demand Predictor", layout="centered")

# Title and image
st.title("🔋 Hourly Electricity Demand Predictor")
st.image("https://static.vecteezy.com/system/resources/thumbnails/024/352/164/small_2x/energy-consumption-and-co2-gas-emissions-are-increasing-light-bulbs-with-green-eco-city-renewable-energy-by-2050-carbon-neutral-energy-save-energy-creative-idea-concept-generative-ai-free-photo.jpg")

st.sidebar.header("🕒 Input Time Features")

# Sidebar inputs
year = st.sidebar.number_input("Year", min_value=2004, max_value=2050, value=2025)
month = st.sidebar.slider("Month", 1, 12, 6)
dayofweek = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
hour = st.sidebar.slider("Hour", 0, 23, 12)
quarter = (month - 1) // 3 + 1
is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])
lag_1 = st.sidebar.number_input("Lag 1 Hour Demand", value=35000.0)
lag_24 = st.sidebar.number_input("Lag 24 Hour Demand", value=36000.0)
rolling_mean_24 = st.sidebar.number_input("Rolling Mean (24h)", value=35500.0)
rolling_std_24 = st.sidebar.number_input("Rolling Std (24h)", value=500.0)

# Prepare input
input_df = pd.DataFrame({
    "Year": [year],
    "Quarter": [quarter],
    "Month": [month],
    "DayOfWeek": [dayofweek],
    "is_weekend": [is_weekend],
    "Hour": [hour],
    "lag_1": [lag_1],
    "lag_24": [lag_24],
    "rolling_mean_24": [rolling_mean_24],
    "rolling_std_24": [rolling_std_24]
})

# Prediction
if st.sidebar.button("Predict Demand"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("🔮 Predicted Electricity Demand")
        st.success(f"{prediction:,.2f} MW")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
