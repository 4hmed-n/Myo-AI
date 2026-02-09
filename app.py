import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Setup
st.set_page_config(page_title="Myo AI Simulator", layout="centered")
st.title("🫀 Myo AI: Cardiovascular Risk Simulator")
st.caption("Powered by Myo-Core Engine (Fusion Model)")

# 2. Load the Model
try:
    model = joblib.load('myocore_pipeline.pkl')
    st.success("System Online: Neural Link Established")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Patient Vitals")

# Clinical
age = st.sidebar.slider("Age (Years)", 20, 100, 50)
gender_opt = st.sidebar.radio("Sex", ["Male", "Female"])
gender = 2 if gender_opt == "Male" else 1

restingbp = st.sidebar.slider("Resting BP (Systolic)", 90, 200, 120)
ap_hi = restingbp
ap_lo = st.sidebar.slider("Diastolic BP", 50, 130, 80)

chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)
chol_cat = 1 if chol < 200 else (2 if chol < 240 else 3)

fbs_opt = st.sidebar.radio("Fasting BS > 120 mg/dl?", ["No", "Yes"])
fastingbs = 1 if fbs_opt == "Yes" else 0
gluc = 1 if fastingbs == 0 else 2

maxhr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)

# Physical
height = st.sidebar.slider("Height (cm)", 100, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 75)
bmi = weight / ((height/100)**2)
smoke = st.sidebar.checkbox("Smoker?")
alco = st.sidebar.checkbox("Alcohol Intake?")
active = st.sidebar.checkbox("Physically Active?")

# 4. Run Simulation
if st.button("Run Simulation"):
    pulse_pressure = ap_hi - ap_lo
    heartdisease = 0 
    
    # EXACT order from your debug log
    columns = [
        'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak', 'heartdisease',
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'ecg_mean', 'ecg_std', 'ecg_skew', 'ecg_kurtosis',
        'sensor_signal_available', 'bmi', 'pulse_pressure'
    ]
    
    row = [
        age, restingbp, chol_cat, fastingbs, maxhr, oldpeak, heartdisease,
        age, gender, height, weight, ap_hi, ap_lo, chol_cat, gluc,
        int(smoke), int(alco), int(active), 0.0, 1.0, 0.0, 3.0, # Defaults for ECG
        1, bmi, pulse_pressure
    ]
    
    input_df = pd.DataFrame([row], columns=columns)

    try:
        prob = model.predict_proba(input_df)[0][1]
        st.subheader("Result")
        st.metric("Risk Probability", f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")
        st.progress(float(prob))
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
