from unittest.mock import patch

from fastapi.testclient import TestClient


class FakeModel:
    def predict_proba(self, data):
        return [[0.8, 0.2]]


with patch("mlflow.sklearn.load_model", return_value=FakeModel()):
    from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {
        "amount": 120.0,
        "transaction_hour": 14,
        "merchant_category": "Clothing",
        "foreign_transaction": 0,
        "location_mismatch": 0,
        "device_trust_score": 85,
        "velocity_last_24h": 1,
        "cardholder_age": 40
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "threshold" in result
    assert "prediction" in result
    assert "result" in result