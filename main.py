import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
@st.cache_data(ttl=10800)
def load_and_preprocess_data(set_path):
    df = pd.read_csv("heart.csv")
    irrelevant_columns = ["id", "dataset"]
    df = df.drop(columns=[col for col in irrelevant_columns if col in df.columns], errors="ignore")
    df.fillna(df.median(numeric_only=True), inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Error: Column '{target_col}' not found in the dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns.tolist(), scaler

@st.cache_data(ttl=10800)
def train_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "heart_disease_model.pkl")
    return model

# ----------------------------
# Prediction and Risk Analysis
# ----------------------------
def assess_risk_factors(user_data):
    recommendations = []
    if user_data["chol"] > 240:
        recommendations.append("Reduce cholesterol by eating more fiber, reducing saturated fats, and exercising regularly.")
    if user_data["trestbps"] > 130:
        recommendations.append("Lower blood pressure through a low-sodium diet, regular exercise, and stress management.")
    if user_data["thalach"] < 100:
        recommendations.append("Increase cardiovascular activity to improve heart rate performance.")
    if not recommendations:
        recommendations.append("Maintain a balanced diet, stay active, and monitor your health regularly.")
    return recommendations

def predict_heart_disease(user_data, model, scaler, feature_names):
    input_array = np.array([user_data[feature] for feature in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    risk = "High" if prediction == 1 else "Low"
    recommendations = assess_risk_factors(user_data)
    return "Heart Disease Detected" if prediction == 1 else "No Heart Disease", risk, recommendations

# ----------------------------
# Dummy AI Assistant (Local Response)
# ----------------------------
def get_local_response(query, user_data):
    context = ", ".join([f"{key}: {value}" for key, value in user_data.items()])
    if "cure" in query.lower():
        return f"Based on your health details ({context}), it is important to consult a cardiologist. Maintain a healthy lifestyle with a good diet, regular exercise, and routine check-ups."
    elif "symptom" in query.lower():
        return "Common heart disease symptoms include chest pain, shortness of breath, fatigue, and irregular heartbeat."
    elif "prevent" in query.lower():
        return "Prevention includes a healthy diet, regular exercise, stress management, avoiding smoking, and managing conditions like hypertension."
    else:
        return "Please consult a certified medical professional for detailed advice."

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction and Risk Analysis")
st.markdown("---")

file_path = "heart.csv"
X_scaled, y, feature_names, scaler = load_and_preprocess_data(file_path)
model = train_model(X_scaled, y)

st.header("Enter Your Health Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[("None", 0), ("Mild", 1), ("Moderate", 2), ("Severe", 3)], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar (>120 mg/dL)", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with col2:
    restecg = st.selectbox("ECG Results", options=[("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)], format_func=lambda x: x[0])[1]
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia Type", options=[("Normal", 0), ("Fixed Defect", 1), ("Reversible Defect", 2)], format_func=lambda x: x[0])[1]

input_data = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}

if st.button("Predict Heart Disease"):
    prediction, risk_level, recommendations = predict_heart_disease(input_data, model, scaler, feature_names)
    st.markdown("---")
    st.header("Prediction Result")
    if prediction == "Heart Disease Detected":
        st.error(f"**Prediction:** {prediction}")
    else:
        st.success(f"**Prediction:** {prediction}")
    st.warning(f"**Risk Level:** {risk_level}")
    st.header("Health Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")
    st.markdown("---")

# ----------------------------
# Local Health Assistant
# ----------------------------
st.header("Personalized Health Assistant")
user_query = st.text_area("Ask a health-related question:")

if user_query.strip():
    response = get_local_response(user_query, input_data)
    st.subheader("Medical Assistant's Advice:")
    st.write(response)
else:
    st.info("Please enter a question for personalized health advice.")
