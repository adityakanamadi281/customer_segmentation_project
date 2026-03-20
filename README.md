# 🧠 Customer Segmentation API

This project performs **customer segmentation** using **K-Means clustering** to group customers based on their purchasing behavior and demographics.  
It provides a robust **FastAPI backend** for interacting with the trained model and predicting customer segments via a REST API. The entire application is fully containerized using **Docker** for easy deployment.

---

## 🚀 Project Overview

Customer segmentation helps businesses understand different groups of customers based on key features like **Age**, **Income**, **Spending habits**, and **Recency** (how recently they purchased).  
By clustering customers, businesses can create targeted marketing strategies, personalized offers, and better customer experiences.

### **1. Data & Features**
The model clusters customers based on the following key features carefully extracted and scaled during preprocessing:
- `Age`
- `Income`
- `TotalSpend`
- `NumWebPurchases`
- `NumStorePurchases`
- `NumWebVisitsMonth`
- `Recency`

### **2. Technical Stack**
- **Language**: Python 3.11+
- **Machine Learning**: `scikit-learn` (K-Means Clustering, StandardScaler)
- **API Framework**: `FastAPI` & `Uvicorn`
- **Data Validation & Handling**: `pydantic`, `pandas`, `numpy`
- **Containerization**: `Docker`

---

## 🛠️ Setup and Installation

### Local Setup (Without Docker)

1. **Clone the repository and navigate to the project directory:**
    ```bash
    git clone <your-repo-url>
    cd customer_segmentation_project
    ```

2. **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    
    # On Windows:
    .venv\Scripts\activate
    
    # On macOS/Linux:
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

3. **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will start locally at `http://127.0.0.1:8000`.

### Docker Setup (Recommended)

Ensure you have **Docker Desktop** installed and running on your device before proceeding.

1. **Build the Docker image:**
    ```bash
    docker build -t customer_segmentation .
    ```

2. **Run the container:**
    ```bash
    docker run -p 8000:8000 customer_segmentation
    ```
    The API will be available at `http://localhost:8000`.

---

## 📖 API Documentation & Usage

FastAPI automatically generates beautiful interactive documentation. Once your server is up and running, you can interact with the API directly from your browser:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Predicting a Segmentation Cluster

You can test the **POST `/predict`** endpoint through the Swagger UI, or run a `curl` request in your terminal:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Age": 45,
  "Income": 60000,
  "TotalSpend": 1200,
  "NumWebPurchases": 5,
  "NumStorePurchases": 8,
  "NumWebVisitsMonth": 3,
  "Recency": 10
}'
```

**Response Example:**
The API will return the predicted integer cluster for the customer.
```json
{
  "segment": 2
}
```

---

## 📂 Project Structure

```
│── Dockerfile              # Instructions for building the container
│── main.py                 # Core FastAPI application logic
│── requirements.txt        # Python dependencies
│── README.md               # You are here
└── models/                 # Directory containing trained joblib models
    ├── kmeans_model.pkl    # Trained KMeans Clustering model
    └── scaler.pkl          # Standardization scaler instance
```
