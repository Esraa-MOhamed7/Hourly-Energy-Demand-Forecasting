# 🔋 Hourly Energy Consumption Forecasting

Welcome to a full-cycle data science project where we analyze and forecast electricity demand using the PJME dataset. Our mission: to understand how energy consumption fluctuates over time—and to build accurate, interpretable models that predict future demand.

---

## Project Workflow

We follow a complete and modular data science pipeline:

### 1. Exploration
We begin by inspecting the dataset using `.head()` and `.shape` to understand its structure, size, and granularity.

### 2. Cleaning
We handle missing values, convert datetime formats, and reset the index to prepare the data for time-based modeling.

### 3. Feature Extraction
We engineer powerful time features:
- `Month`, `Year`, `Hour`, `DayOfWeek`, `Quarter`  
These help capture seasonal, weekly, and hourly demand patterns.

### 4. Analysis & Visualization
We explore demand trends using expressive plots:
- Line charts, boxplots, and seasonal breakdowns  
These visualizations reveal cyclical behavior and outliers.

### 5. Modeling
We train and compare multiple machine learning models:
- Linear Regression  
- Ridge Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost

**XGBoost** and **Random Forest** achieved the highest accuracy with **R² = 0.99**.

---

## Web App Overview

The deployed Streamlit app allows users to:
- Input time-based features interactively  
- Predict electricity demand in megawatts  
- View model performance and feature importance

The model is hosted on Google Drive and downloaded dynamically using `gdown`, keeping the app lightweight and reproducible.

---

##  Performance Metrics

We measure performance using:

- **MAE** (Mean Absolute Error): `367.93`  
- **RMSE** (Root Mean Squared Error): `608.04`  
- **R² Score**: `0.9911`

✅ These results reflect high model accuracy, especially from XGBoost and Random Forest.

---

## Charts
Below are examples of visualizations used in the project and app:

![chart1](https://github.com/Esraa-MOhamed7/Hourly-Energy-Demand-Forecasting/blob/main/PJME%20Electricity%20Demand%20Distribution%20by%20Quarter.png)
![chart1](https://github.com/Esraa-MOhamed7/Hourly-Energy-Demand-Forecasting/blob/main/Quarter%20Distribution%20of%20PJME%20Electricity%20Demand.png)
![chart1](dss.ff)
![chart1](dss.ff)
![chart1](dss.ff)
![chart1](dss.ff)

