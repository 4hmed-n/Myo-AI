import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# Check versions
st.sidebar.caption(f"scikit-learn: {sklearn.__version__}")

# FIX: Monkey patch for SimpleImputer compatibility between sklearn versions
# This handles the '_fill_dtype' attribute error
from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = property(lambda self: None)

# 1. Page Setup
st.set_page_config(page_title="Myo AI Simulator", layout="centered")
st.title("🫀 Myo AI: Cardiovascular Risk Simulator")
st.caption("Powered by Myo-Core Engine (Fusion Model)")

# 2. Load the Model
try:
    model = joblib.load('myocore_pipeline.pkl')
    st.success("System Online: Neural Link Established")
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please upload 'myocore_pipeline.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Patient Vitals")

# --- Clinical Features ---
age = st.sidebar.slider("Age (Years)", 20, 100, 50)

gender_opt = st.sidebar.radio("Sex", ["Male", "Female"])
sex = 1 if gender_opt == "Male" else 0
gender = 2 if gender_opt == "Male" else 1

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)

restingbp = st.sidebar.slider("Resting BP (mm Hg) - Systolic", 90, 200, 120)
ap_hi = restingbp
ap_lo = st.sidebar.slider("Diastolic BP (mm Hg)", 50, 130, 80)

chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)
cholesterol_cat = 1 if chol < 200 else (2 if chol < 240 else 3)

fbs_opt = st.sidebar.radio("Fasting BS > 120 mg/dl?", ["No", "Yes"])
fastingbs = 1 if fbs_opt == "Yes" else 0
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

bmi = weight / ((height/100)**2)

smoke = st.sidebar.checkbox("Smoker?")
alco = st.sidebar.checkbox("Alcohol Intake?")
active = st.sidebar.checkbox("Physically Active?")

# --- ECG Signal Features ---
st.sidebar.markdown("---")
st.sidebar.header("ECG Signal Parameters")
ecg_mean = st.sidebar.slider("ECG Mean", -5.0, 5.0, 0.0)
ecg_std = st.sidebar.slider("ECG Std Dev", 0.0, 5.0, 1.0)
ecg_skew = st.sidebar.slider("ECG Skewness", -3.0, 3.0, 0.0)
ecg_kurtosis = st.sidebar.slider("ECG Kurtosis", 0.0, 10.0, 3.0)

# 4. Run Simulation
if st.button("Run Simulation"):
    
    pulse_pressure = ap_hi - ap_lo
    heartdisease = 0  # Placeholder as required by model
    
    # Build DataFrame with exact column order
    expected_order = [
        'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak', 'heartdisease',
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'ecg_mean', 'ecg_std', 'ecg_skew', 'ecg_kurtosis',
        'sensor_signal_available', 'bmi', 'pulse_pressure'
    ]
    
    values = [
        age, restingbp, cholesterol_cat, fastingbs, maxhr, oldpeak, heartdisease,
        age, gender, height, weight, ap_hi, ap_lo, cholesterol_cat, gluc,
        int(smoke), int(alco), int(active), ecg_mean, ecg_std, ecg_skew, ecg_kurtosis,
        1, bmi, pulse_pressure
    ]
    
    input_data = pd.DataFrame([values], columns=expected_order)

    try:
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
            st.caption(f"BMI: {bmi:.1f} | Pulse Pressure: {pulse_pressure:.0f} mmHg")
        
        with col2:
            if probability > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
                st.progress(float(probability))
            else:
                st.success("✅ LOW RISK DETECTED")
                st.progress(float(probability))
                
        # Additional details
        with st.expander("Input Data Verification"):
            st.write("Features provided:", input_data.shape[1])
            st.dataframe(input_data.T.rename(columns={0: 'Value'}))
                
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Try upgrading scikit-learn: `pip install --upgrade scikit-learn`")
