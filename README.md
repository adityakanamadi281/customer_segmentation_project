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

