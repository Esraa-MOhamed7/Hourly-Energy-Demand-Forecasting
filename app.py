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

# App config
st.set_page_config(page_title="🔋 Electricity Demand Predictor", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: teal;'>🔋 Hourly Electricity Demand Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Welcome! Enter time features to predict electricity demand.")

# Image banner
st.image("https://static.vecteezy.com/system/resources/thumbnails/024/352/164/small_2x/energy-consumption-and-co2-gas-emissions-are-increasing-light-bulbs-with-green-eco-city-renewable-energy-by-2050-carbon-neutral-energy-save-energy-creative-idea-concept-generative-ai-free-photo.jpg")

# Tabs
tab1, tab2, tab3 = st.tabs([" Prediction", " Trends", " Model Info"])

with tab1:
    st.sidebar.header("🕒 Input Time Features")
    year = st.sidebar.number_input("Year", min_value=2004, max_value=2050, value=2025, help="Enter any year (2004–2050)")
    month = st.sidebar.slider("Month", 1, 12, 6, help="Month of the year")
    dayofweek = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2, help="0 = Monday, 6 = Sunday")
    hour = st.sidebar.slider("Hour", 0, 23, 12, help="Hour of the day (0 = midnight)")
    quarter = (month - 1) // 3 + 1  # Auto-calculated from month
    is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])
    lag_1 = st.sidebar.number_input("Lag 1 Hour Demand", value=35000.0, help="Previous hour's demand (typical range: 30,000–45,000 MW)")
    lag_24 = st.sidebar.number_input("Lag 24 Hour Demand", value=36000.0, help="Demand 24 hours ago")
    rolling_mean_24 = st.sidebar.number_input("Rolling Mean (24h)", value=35500.0, help="Average demand over past 24 hours")
    rolling_std_24 = st.sidebar.number_input("Rolling Std (24h)", value=500.0, help="Standard deviation over past 24 hours")

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
        with st.spinner("Calculating..."):
            prediction = model.predict(input_df)[0]
        st.subheader(" Predicted Electricity Demand")
        st.metric(label="PJME_MW", value=f"{prediction:,.2f}")
        st.markdown(f"<h4 style='color: green;'> Estimated Demand: {prediction:,.2f} MW</h4>", unsafe_allow_html=True)

with tab2:
    st.subheader(" Sample Demand Trend")
    try:
        sample_data = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
        st.line_chart(sample_data["PJME_MW"].tail(500))
        with st.expander("📂 View Sample Data"):
            st.dataframe(sample_data.tail(100))
    except:
        st.warning("⚠️ Sample data not found. Please add PJME_hourly.csv to your folder.")

with tab3:
    st.subheader("Model Performance")
    st.success("R² Score: 0.9911 (based on test data)")
    st.info("RMSE: 609.31 MW")

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    features = input_df.columns
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax)
    st.pyplot(fig)