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
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please upload 'myocore_pipeline.pkl' to GitHub.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("If you see a version error, make sure requirements.txt contains 'scikit-learn==1.3.2'")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Patient Vitals")

# --- Clinical Features ---
age = st.sidebar.slider("Age (Years)", 20, 100, 50)

gender_opt = st.sidebar.radio("Sex", ["Male", "Female"])
# Dual encoding to satisfy all naming conventions
sex = 1 if gender_opt == "Male" else 0
gender = 2 if gender_opt == "Male" else 1

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)

# Blood Pressure
restingbp = st.sidebar.slider("Resting BP (Systolic)", 90, 200, 120)
ap_hi = restingbp
ap_lo = st.sidebar.slider("Diastolic BP", 50, 130, 80)

chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)
# Categorical mapping
cholesterol_cat = 1 if chol < 200 else (2 if chol < 240 else 3)

fbs_opt = st.sidebar.radio("Fasting BS > 120 mg/dl?", ["No", "Yes"])
fastingbs = 1 if fbs_opt == "Yes" else 0
# Glucose mapping
gluc = 1 if fastingbs == 0 else 2

restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2], index=0)
maxhr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

exng_opt = st.sidebar.radio("Exercise Induced Angina?", ["No", "Yes"])
exang = 1 if exng_opt == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope", [0, 1, 2], index=1)
caa = st.sidebar.slider("Major Vessels (0-4)", 0, 4, 0)
thall = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3], index=2)

# --- Lifestyle Features ---
st.sidebar.markdown("---")
st.sidebar.header("Lifestyle & Physical")
height = st.sidebar.slider("Height (cm)", 100, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 75)

# BMI Calculation
bmi = weight / ((height/100)**2)

smoke = st.sidebar.checkbox("Smoker?")
alco = st.sidebar.checkbox("Alcohol Intake?")
active = st.sidebar.checkbox("Physically Active?")

# --- ECG Signal Features (Hidden Defaults) ---
# We keep these hidden or defaulted to avoid overwhelming the user
ecg_mean = 0.0
ecg_std = 1.0
ecg_skew = 0.0
ecg_kurtosis = 3.0

# 4. Run Simulation
if st.button("Run Simulation"):
    
    # Calculated Fields
    pulse_pressure = ap_hi - ap_lo
    heartdisease = 0  # Dummy target variable (required by some pipeline scalers)
    
    # EXACT COLUMN ORDER (Crucial for Scikit-Learn pipelines)
    # This matches the error logs we saw earlier
    expected_order = [
        'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak', 'heartdisease',
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'ecg_mean', 'ecg_std', 'ecg_skew', 'ecg_kurtosis',
        'sensor_signal_available', 'bmi', 'pulse_pressure'
    ]
    
    # Values mapped to the order above
    values = [
        age, restingbp, cholesterol_cat, fastingbs, maxhr, oldpeak, heartdisease,
        age, gender, height, weight, ap_hi, ap_lo, cholesterol_cat, gluc,
        int(smoke), int(alco), int(active), ecg_mean, ecg_std, ecg_skew, ecg_kurtosis,
        1, bmi, pulse_pressure
    ]
    
    # Create DataFrame
    input_data = pd.DataFrame([values], columns=expected_order)

    try:
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
            st.caption(f"BMI: {bmi:.1f} | Pulse Pressure: {pulse_pressure:.0f}")
        
        with col2:
            if probability > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
            else:
                st.success("✅ LOW RISK DETECTED")
            st.progress(float(probability))
                
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug info - Expected columns:", expected_order)
