from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_version():
    response = client.get("/model/version")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "RandomForestClassifier"
    assert data["version"] == "1.0"


def test_fraud_prediction():
    transaction = {
        "amount": 95000,
        "transaction_hour": 2,
        "device_score": 0.10,
        "account_age": 3,
        "previous_transactions": 1,
        "location_score": 0.15
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "FRAUD"
    assert 0 <= data["fraud_probability"] <= 1


def test_genuine_prediction():
    transaction = {
        "amount": 1200,
        "transaction_hour": 14,
        "device_score": 0.90,
        "account_age": 60,
        "previous_transactions": 15,
        "location_score": 0.90
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "GENUINE"
    assert 0 <= data["fraud_probability"] <= 1