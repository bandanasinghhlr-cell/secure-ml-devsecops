import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
DATA_PATH = "ml/data/transactions.csv"
MODEL_PATH = "ml/model/fraud_model.pkl"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print(f"Dataset shape: {df.shape}")


# 2. Separate features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# 5. Train model
model.fit(X_train, y_train)

print("Model training completed")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 8. Save trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")