import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('diabetes_lr_model.joblib')
scaler = joblib.load('scaled.joblib')

st.title("🩺 Diabetes Prediction")

st.write("Enter patient information to predict diabetes risk:")

# 1️⃣ User Inputs
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1)

# 2️⃣ Predict Button
if st.button("Predict"):

    # 3️⃣ Feature Engineering
    has_pregnancies = 1 if pregnancies > 0 else 0
    glucose_bmi = glucose * bmi

    # BMI category one-hot
    bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    bmi_cat_values = [0, 0, 0, 0]
    if bmi < 18.5:
        bmi_cat_values[0] = 1
    elif bmi < 25:
        bmi_cat_values[1] = 1
    elif bmi < 30:
        bmi_cat_values[2] = 1
    else:
        bmi_cat_values[3] = 1

    # Age group one-hot
    age_groups = ['21s', '31s', '41s', '51s', '60+']
    age_group_values = [0, 0, 0, 0, 0]
    if age <= 30:
        age_group_values[0] = 1
    elif age <= 40:
        age_group_values[1] = 1
    elif age <= 50:
        age_group_values[2] = 1
    elif age <= 60:
        age_group_values[3] = 1
    else:
        age_group_values[4] = 1

    # 4️⃣ Combine all features (MUST match model training order)
    features = [
        pregnancies, glucose, blood_pressure, skin_thickness, insulin,
        bmi, dpf, age,
        has_pregnancies, glucose_bmi
    ]
    features += bmi_cat_values + age_group_values

    features = np.array([features])
    features_scaled = scaler.transform(features)

    # 5️⃣ Predict
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    # 6️⃣ Display Result
    if prediction == 1:
        st.error(f"🚨 Prediction: Diabetic (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Prediction: Not Diabetic (Confidence: {prob:.2%})")
