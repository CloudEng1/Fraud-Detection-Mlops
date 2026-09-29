from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd


# -----------------------------
# MLflow configuration
# -----------------------------
MODEL_PATH = "artifacts/fraud_model"

model = mlflow.sklearn.load_model(MODEL_PATH)


# -----------------------------
# FastAPI application
# -----------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="ML API for detecting potentially fraudulent transactions",
    version="1.0.0"
)


# -----------------------------
# Request schema
# -----------------------------
class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running 🚀",
        "status": "healthy"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([transaction.model_dump()])

    probability = model.predict_proba(data)[0][1]

    threshold = 0.30

    prediction = int(probability >= threshold)

    result = "Fraud" if prediction == 1 else "Legitimate"

    return {
        "fraud_probability": round(float(probability), 4),
        "threshold": threshold,
        "prediction": prediction,
        "result": result
    }