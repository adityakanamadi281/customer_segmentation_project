import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load(r"C:\Users\adity\customer_segmentation_project\models\kmeans_model.pkl")
scaler = joblib.load(r"C:\Users\adity\customer_segmentation_project\models\scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter a customer details to predict the segment")

#"Age", "Income", 'TotalSpend', 'NumWebPurchases' , 'NumStorePurchases',  'NumWebVisitsMonth', 'Recency'
age = st.number_input("Age", min_value=18,max_value=100)
income = st.number_input("Income", min_value=0, max_value=200000)
total_spend = st.number_input("TotalSpend", min_value=0, max_value=5000)
num_web_purchases = st.number_input("NumWebPurchases", min_value=0, max_value=100)
num_store_purchases = st.number_input("NumStorePurchases", min_value=0, max_value=100)
num_web_visits_month = st.number_input("NumWebVisitsMonth",min_value=0, max_value=50)
recency = st.number_input("Recency",min_value=0, max_value=365)



input_data = pd.DataFrame(
    {
        "Age":[age],
        "Income":[income],
        "TotalSpend" : [total_spend],
        "NumWebPurchases" : [num_web_purchases],
        "NumStorePurchases" : [num_store_purchases],
        "NumWebVisitsMonth" : [num_web_visits_month],
        "Recency" : [recency]
    }
)

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment "):
    clusters = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment : Cluster {clusters}")
    
               

