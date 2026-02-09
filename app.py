import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Setup
st.set_page_config(page_title="Myo AI Simulator", layout="centered")
st.title("🫀 Myo AI: Cardiovascular Risk Simulator")
st.caption("Powered by Myo-Core Engine")

# 2. Load the Model
try:
    model = joblib.load('myocore_pipeline.pkl')
    st.success("System Online: Neural Link Established")
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please upload 'myocore_pipeline.pkl' to GitHub.")
    st.stop()

# 3. Sidebar Inputs (MATCHING YOUR DATASET EXACTLY)
st.sidebar.header("Patient Vitals")

# Age (Input in years, converted to days if model needs it, but usually years)
age_years = st.sidebar.slider("Age (Years)", 20, 100, 50)
age = age_years # Assuming model uses years. If trained on days, change to: age_years * 365

# Gender
gender_option = st.sidebar.radio("Gender", ["Female", "Male"])
gender = 2 if gender_option == "Male" else 1  # 1: Female, 2: Male (Common coding for this dataset)

# Physical Traits (needed for BMI)
height = st.sidebar.slider("Height (cm)", 100, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 75)

# Blood Pressure
ap_hi = st.sidebar.slider("Systolic BP (ap_hi)", 90, 220, 120)
ap_lo = st.sidebar.slider("Diastolic BP (ap_lo)", 60, 140, 80)

# Lab Results
chol_option = st.sidebar.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
cholesterol = 1 if chol_option == "Normal" else (2 if chol_option == "Above Normal" else 3)

gluc_option = st.sidebar.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
gluc = 1 if gluc_option == "Normal" else (2 if gluc_option == "Above Normal" else 3)

# Lifestyle
smoke = st.sidebar.checkbox("Smoker?")
alco = st.sidebar.checkbox("Alcohol Intake?")
active = st.sidebar.checkbox("Physically Active?")

# Convert booleans to 0/1
smoke_val = 1 if smoke else 0
alco_val = 1 if alco else 0
active_val = 1 if active else 0

# 4. Feature Engineering (Calculating BMI)
# Your error showed 'bmi' is missing, so we must calculate it!
bmi = weight / ((height / 100) ** 2)

# 5. Prediction Logic
if st.button("Run Simulation"):
    # Create DataFrame with EXACT column names from your error message
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke_val],
        'alco': [alco_val],
        'active': [active_val],
        'bmi': [bmi]  # The engineered feature
    })

    try:
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Display
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
            st.caption(f"Calculated BMI: {bmi:.1f}")
        
        with col2:
            if probability > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
            else:
                st.success("✅ LOW RISK DETECTED")
                
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Debug - Your model expects these columns:", model.feature_names_in_)
