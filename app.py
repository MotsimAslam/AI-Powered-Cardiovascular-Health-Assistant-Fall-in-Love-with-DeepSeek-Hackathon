# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, names=column_names, na_values="?")
    data = data.dropna()
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
    return data

data = load_data()

# Preprocess the data
X = data.drop("target", axis=1)
y = data["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train an XGBoost model
@st.cache_resource
def train_model():
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

model = train_model()

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

# Streamlit App
st.title("Cardiovascular Disease (CVD) Risk Assessment")

# Tabs
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Health Coach", "Remote Monitoring"])

# Risk Assessment Tab
with tab1:
    st.header("Risk Assessment")
    
    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.radio("Gender", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (1-4)", min_value=1, max_value=4, value=1)
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
    fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dL")
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, value=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.checkbox("Exercise-Induced Angina")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope of the Peak Exercise ST Segment (1-3)", min_value=1, max_value=3, value=1)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", min_value=0, max_value=3, value=0)
    thal = st.number_input("Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)", min_value=3, max_value=7, value=3)
    smoker = st.checkbox("Do you smoke?")
    
    # Prepare input for the model
    input_data = [[age, 1 if sex == "Male" else 0, cp, trestbps, chol, 1 if fbs else 0, restecg, thalach, 1 if exang else 0, oldpeak, slope, ca, thal]]
    input_scaled = scaler.transform(input_data)
    
    # Button to calculate risk
    if st.button("Calculate Risk"):
        risk_prediction = model.predict(input_scaled)[0]
        risk_probability = model.predict_proba(input_scaled)[0][1]
        
        # Display the result
        st.subheader("Your CVD Risk Prediction")
        if risk_prediction == 1:
            st.error(f"High Risk: You have a {risk_probability * 100:.2f}% chance of having heart disease. Please consult a doctor.")
        else:
            st.success(f"Low Risk: You have a {risk_probability * 100:.2f}% chance of having heart disease. Keep up the good habits!")
        
        # Plotly Visualization
        st.subheader("Risk Probability Distribution")
        fig = px.histogram(data, x="age", color="target", nbins=20, labels={"age": "Age", "target": "Heart Disease"})
        st.plotly_chart(fig)
        
        # Health coach feedback
        feedback = health_coach_feedback(age, chol, trestbps, smoker)
        if feedback:
            st.subheader("Health Coach Feedback")
            for message in feedback:
                st.write(message)
        
        # Anomaly alerts
        alerts = anomaly_alerts(trestbps, chol)
        if alerts:
            st.subheader("Anomaly Alerts")
            for alert in alerts:
                st.error(alert)

# Health Coach Tab
with tab2:
    st.header("Health Coach")
    
    # Diet logs
    st.subheader("Diet Logs")
    food_log = st.text_area("What did you eat today?")
    if st.button("Submit Diet Log"):
        st.success("Diet log submitted successfully!")
    
    # Feedback based on diet
    if food_log:
        if "junk" in food_log.lower():
            st.warning("You consumed junk food. Consider healthier options.")
        else:
            st.success("Your diet looks healthy. Keep it up!")

# Remote Monitoring Tab
with tab3:
    st.header("Remote Monitoring")
    
    # Simulate real-time alerts
    st.subheader("Real-Time Alerts")
    if st.button("Check for Alerts"):
        if trestbps > 180 or chol > 300:
            st.error("Critical Alert: High blood pressure or cholesterol detected. Seek medical attention!")
        else:
            st.success("No critical alerts detected. Your vitals are within normal range.")
