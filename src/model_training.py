import pandas as pd
import numpy as np
import os
from pathlib import Path
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
import joblib



PROCESSED_PATH = Path(r"C:\Users\adity\customer_segmentation_project\data\processed\cleaned_data.csv")
df = pd.read_csv(PROCESSED_PATH)


features = ["Age", "Income", 'TotalSpend', 'NumWebPurchases' , 'NumStorePurchases',  'NumWebVisitsMonth', 'Recency']

x=df[features].copy()

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

wcss = []

for i in range(2,10):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)


joblib.dump(kmeans, r"C:\Users\adity\customer_segmentation_project\models\kmeans_model.pkl")
joblib.dump(scaler, r"C:\Users\adity\customer_segmentation_project\models\scaler.pkl")

