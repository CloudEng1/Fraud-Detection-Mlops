import os
import tempfile

import boto3
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


S3_BUCKET = os.getenv("MODEL_S3_BUCKET")
S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "fraud_model/")

LOCAL_MODEL_PATH = "artifacts/fraud_model"


def download_model_from_s3():
    model_dir = os.path.join(tempfile.gettempdir(), "fraud_model")
    os.makedirs(model_dir, exist_ok=True)

    s3 = boto3.client("s3")

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX
    )

    if "Contents" not in response:
        raise RuntimeError(
            f"No model files found in s3://{S3_BUCKET}/{S3_PREFIX}"
        )

    for obj in response["Contents"]:
        key = obj["Key"]

        if key.endswith("/"):
            continue

        relative_path = key[len(S3_PREFIX):]
        local_path = os.path.join(model_dir, relative_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3.download_file(
            S3_BUCKET,
            key,
            local_path
        )

    return model_dir


if S3_BUCKET:
    MODEL_PATH = download_model_from_s3()
else:
    MODEL_PATH = LOCAL_MODEL_PATH


model = mlflow.sklearn.load_model(MODEL_PATH)


app = FastAPI(
    title="Fraud Detection API",
    description="ML API for detecting potentially fraudulent transactions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running 🚀",
        "status": "healthy"
    }


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