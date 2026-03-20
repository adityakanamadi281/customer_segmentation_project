from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Segmentation API")

# Setup correct file paths for docker and local execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load models
try:
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    kmeans = None
    scaler = None

class CustomerData(BaseModel):
    Age: int
    Income: float
    TotalSpend: float
    NumWebPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int
    Recency: int

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Segmentation API"}

@app.post("/predict")
def predict_segment(data: CustomerData):
    if kmeans is None or scaler is None:
        return {"error": "Model not loaded"}
    
    # Prepare data for prediction
    input_data = pd.DataFrame([data.model_dump()])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]
    
    return {"segment": int(cluster)}
