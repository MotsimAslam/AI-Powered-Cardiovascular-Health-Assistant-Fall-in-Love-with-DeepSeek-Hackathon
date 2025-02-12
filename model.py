
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, names=column_names, na_values="?")
    data = data.dropna()
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
    return data

# Preprocess the data
def preprocess_data(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Train the XGBoost model
def train_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

# Calculate risk score
def calculate_risk(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    risk_prediction = model.predict(input_scaled)[0]
    risk_probability = model.predict_proba(input_scaled)[0][1]
    return risk_prediction, risk_probability

# Health Coach Feedback
def health_coach_feedback(age, cholesterol, blood_pressure, smoker):
    feedback = []
    if age > 50:
        feedback.append("You're over 50. Regular health checkups are recommended.")
    if cholesterol > 200:
        feedback.append("Your cholesterol is high. Consider a healthier diet.")
    if blood_pressure > 140:
        feedback.append("Your blood pressure is high. Monitor it regularly.")
    if smoker:
        feedback.append("Smoking increases your risk of heart disease. Consider quitting.")
    return feedback

# Anomaly Alerts
def anomaly_alerts(blood_pressure, cholesterol):
    alerts = []
    if blood_pressure > 180:
        alerts.append("Warning: Extremely high blood pressure detected. Seek medical attention immediately.")
    if cholesterol > 300:
        alerts.append("Warning: Extremely high cholesterol detected. Consult a doctor.")
    return alerts
