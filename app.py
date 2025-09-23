import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

# Download model from Google Drive if not already present
model_url = "https://drive.google.com/uc?id=1Zx_Wbsxgd73U93OOvJ76bT7qOTe7_Gw8"
model_path = "random_forest_model.pkl"

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

# Load model
model = joblib.load(model_path)

# App config (centered layout for mobile compatibility)
st.set_page_config(page_title="🔋 Electricity Demand Predictor", layout="centered")
# Header
st.markdown("<h1 style='text-align: center; color: teal;'>🔋 Hourly Electricity Demand Predictor</h1>", unsafe_allow_html=True)

# Image banner (restored)
st.image("https://static.vecteezy.com/system/resources/thumbnails/024/352/164/small_2x/energy-consumption-and-co2-gas-emissions-are-increasing-light-bulbs-with-green-eco-city-renewable-energy-by-2050-carbon-neutral-energy-save-energy-creative-idea-concept-generative-ai-free-photo.jpg")

# Tabs
tab1, _, tab3 = st.tabs([" Prediction", " Trends (disabled)", " Model Info"])

with tab1:
    st.sidebar.header("🕒 Input Time Features")
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

    if st.button(" Predict Demand"):
        try:
            with st.spinner("Calculating..."):
                prediction = model.predict(input_df)[0]
            st.subheader(" Predicted Electricity Demand")
            st.metric(label="PJME_MW", value=f"{prediction:,.2f}")
            st.markdown(f"<h4 style='color: green;'> Estimated Demand: {prediction:,.2f} MW</h4>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

with tab3:
    st.subheader("Model Performance")
    st.success("R² Score: 0.9911 (based on test data)")
    st.info("RMSE: 609.31 MW")

    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        features = input_df.columns
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Unable to display feature importance: {e}")
