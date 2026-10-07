from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# Load trained ML model
MODEL_PATH = "ml/model/fraud_model.pkl"
model = joblib.load(MODEL_PATH)


# Create FastAPI application
app = FastAPI(
    title="Secure ML Fraud Detection API",
    version="1.0.0"
)


# Request schema
class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    device_score: float
    account_age: int
    previous_transactions: int
    location_score: float


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Model version
@app.get("/model/version")
def model_version():
    return {
        "model": "RandomForestClassifier",
        "version": "1.0"
    }


# Fraud prediction
@app.post("/predict")
def predict(transaction: Transaction):

    data = [[
        transaction.amount,
        transaction.transaction_hour,
        transaction.device_score,
        transaction.account_age,
        transaction.previous_transactions,
        transaction.location_score
    ]]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "FRAUD"
    else:
        result = "GENUINE"

    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 4)
    }