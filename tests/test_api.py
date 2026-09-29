from fastapi.testclient import TestClient

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
    assert "prediction" in result
    assert "result" in result