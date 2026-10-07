import joblib
import pandas as pd


MODEL_PATH = "ml/model/fraud_model.pkl"


def predict_fraud(transaction):
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Convert transaction into DataFrame
    data = pd.DataFrame([transaction])

    # Make prediction
    prediction = model.predict(data)[0]

    # Get probability
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "FRAUD"
    else:
        result = "GENUINE"

    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 4)
    }


if __name__ == "__main__":

    transaction = {
    "amount": 1200,
    "transaction_hour": 14,
    "device_score": 0.90,
    "account_age": 60,
    "previous_transactions": 15,
    "location_score": 0.90
}

    result = predict_fraud(transaction)

    print("Transaction:")
    print(transaction)

    print("\nPrediction:")
    print(result)