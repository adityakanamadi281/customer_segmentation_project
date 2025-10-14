# 🧠 Customer Segmentation using K-Means Clustering

This project performs **customer segmentation** using **K-Means clustering** to group customers based on their purchasing behavior and demographics.  
It also includes a **Streamlit web app** for interactive prediction of customer segments.

---

## 🚀 Project Overview

Customer segmentation helps businesses understand different groups of customers based on key features like **Age**, **Income**, **Spending habits**, and **Recency** (how recently they purchased).  
By clustering customers, businesses can create targeted marketing strategies, personalized offers, and better customer experiences.

---




<img width="1780" height="964" alt="Screenshot 2025-10-13 174539" src="https://github.com/user-attachments/assets/278455b0-d85c-4459-8466-b274a562eae4" />




### **1. Data Preprocessing**
- Load the cleaned dataset.
- Select important features for clustering:
  - `Age`
  - `Income`
  - `TotalSpend`
  - `NumWebPurchases`
  - `NumStorePurchases`
  - `NumWebVisitsMonth`
  - `Recency`
- Standardize features using `StandardScaler`.

### **2. Model Training**
- Apply **K-Means clustering**.
- Tune number of clusters (`n_clusters`) by analyzing **WCSS** (Within-Cluster Sum of Squares).
- Save the trained model and scaler using `joblib`.

### **3. Streamlit Web App**
- Users input customer details through an interface.
- Inputs are scaled using the saved scaler.
- The trained model predicts the customer’s **segment (cluster)**.
- Displays the predicted cluster in real-time.

---

## 🧩 Streamlit App Code Summary

```python
import streamlit as st
import pandas as pd
import joblib

# Load trained models
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Customer Segmentation App")

# User input
age = st.number_input("Age", 18, 100, 35)
income = st.number_input("Income", 0, 200000, 50000)
total_spend = st.number_input("Total Spend", 0, 5000, 1000)
num_web_purchases = st.number_input("Web Purchases", 0, 100, 10)
num_store_purchases = st.number_input("Store Purchases", 0, 100, 10)
num_web_visits_month = st.number_input("Web Visits/Month", 0, 50, 3)
recency = st.number_input("Recency (days)", 0, 365, 30)

# Create input DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "TotalSpend": [total_spend],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits_month],
    "Recency": [recency]
})

# Scale and predict
input_scaled = scaler.transform(input_data)
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")
🧪 Training Script Summary

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import joblib
from pathlib import Path

# Load processed data
df = pd.read_csv(Path("data/processed/cleaned_data.csv"))

# Select features
features = ["Age", "Income", "TotalSpend", "NumWebPurchases",
            "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
X = df[features]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Save models
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
