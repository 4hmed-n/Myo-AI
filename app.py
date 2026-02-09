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
    st.error("⚠️ Model file not found! Please upload 'myocore_pipeline.pkl'.")
    st.stop()

# 3. Sidebar Inputs
st.sidebar.header("Patient Vitals")

# --- Clinical Features (Dataset 1 & 2 mapping) ---
age = st.sidebar.slider("Age (Years)", 20, 100, 50)

gender_opt = st.sidebar.radio("Sex", ["Male", "Female"])
sex = 1 if gender_opt == "Male" else 0  # For Dataset 1
gender = 2 if gender_opt == "Male" else 1  # For Dataset 2 (different encoding!)

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)

# Blood Pressure - both systolic (ap_hi) and diastolic (ap_lo) needed
restingbp = st.sidebar.slider("Resting BP (mm Hg) - Systolic", 90, 200, 120)
ap_hi = restingbp  # Same as restingbp
ap_lo = st.sidebar.slider("Diastolic BP (mm Hg)", 50, 130, 80)  # Added!

chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)
# Map to categorical 1, 2, 3 for Dataset 2
cholesterol_cat = 1 if chol < 200 else (2 if chol < 240 else 3)

fbs_opt = st.sidebar.radio("Fasting BS > 120 mg/dl?", ["No", "Yes"])
fastingbs = 1 if fbs_opt == "Yes" else 0  # Dataset 1 name
gluc = 1 if fastingbs == 0 else 2  # Dataset 2 mapping

restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2], index=0)
maxhr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)  # Changed name

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

# Calculate BMI
bmi = weight / ((height/100)**2)

smoke = st.sidebar.checkbox("Smoker?")
alco = st.sidebar.checkbox("Alcohol Intake?")
active = st.sidebar.checkbox("Physically Active?")

# --- ECG Signal Features (Hidden but required) ---
st.sidebar.markdown("---")
st.sidebar.header("ECG Signal Parameters (Optional)")
ecg_mean = st.sidebar.slider("ECG Mean", -5.0, 5.0, 0.0)
ecg_std = st.sidebar.slider("ECG Std Dev", 0.0, 5.0, 1.0)
ecg_skew = st.sidebar.slider("ECG Skewness", -3.0, 3.0, 0.0)
ecg_kurtosis = st.sidebar.slider("ECG Kurtosis", 0.0, 10.0, 3.0)

# 4. Run Simulation
if st.button("Run Simulation"):
    
    # Calculate derived features
    pulse_pressure = ap_hi - ap_lo  # Now using actual diastolic value
    
    # NOTE: heartdisease is expected by model (even though it's the target!)
    # This suggests the model was trained with target leakage or as a specific feature
    heartdisease = 0  # Default/placeholder value
    
    # Build input DataFrame with EXACT column names and ORDER the model expects
    # Note: The model expects 'age' twice (from different datasets)
    input_data = pd.DataFrame({
        # First set (Dataset 1 - Heart Disease)
        'age': [age],
        'restingbp': [restingbp],
        'cholesterol': [cholesterol_cat],  # Using categorical version
        'fastingbs': [fastingbs],
        'maxhr': [maxhr],
        'oldpeak': [oldpeak],
        'heartdisease': [heartdisease],  # Required by model (unusual but necessary)
        
        # Second set (Dataset 2 - Cardio)
        'age': [age],  # Duplicate - intentional!
        'gender': [gender],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol_cat],  # Duplicate column name!
        'gluc': [gluc],
        'smoke': [1 if smoke else 0],
        'alco': [1 if alco else 0],
        'active': [1 if active else 0],
        
        # ECG Features
        'ecg_mean': [ecg_mean],
        'ecg_std': [ecg_std],
        'ecg_skew': [ecg_skew],
        'ecg_kurtosis': [ecg_kurtosis],
        
        # Derived features
        'sensor_signal_available': [1],  # Always 1 since we have ECG data
        'bmi': [bmi],
        'pulse_pressure': [pulse_pressure],
    })

    # Reorder to match model's expected order exactly
    expected_order = [
        'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak', 'heartdisease',
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'ecg_mean', 'ecg_std', 'ecg_skew', 'ecg_kurtosis',
        'sensor_signal_available', 'bmi', 'pulse_pressure'
    ]
    
    # Handle duplicate column names by creating them separately
    # Pandas doesn't allow duplicates in dict keys, so we build it differently
    
    # Create list of values in exact order
    values = [
        age, restingbp, cholesterol_cat, fastingbs, maxhr, oldpeak, heartdisease,
        age, gender, height, weight, ap_hi, ap_lo, cholesterol_cat, gluc,
        int(smoke), int(alco), int(active), ecg_mean, ecg_std, ecg_skew, ecg_kurtosis,
        1, bmi, pulse_pressure
    ]
    
    # Create DataFrame with duplicate columns properly
    input_data = pd.DataFrame([values], columns=expected_order)

    try:
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
            st.caption(f"Patient BMI: {bmi:.1f}")
            st.caption(f"Pulse Pressure: {pulse_pressure:.0f} mmHg")
        
        with col2:
            if probability > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
            else:
                st.success("✅ LOW RISK DETECTED")
                
    except Exception as e:
        st.error(f"Error: {e}")
        # Debug info
        if hasattr(model, 'feature_names_in_'):
            st.write("Model expects:", list(model.feature_names_in_))
            st.write("You provided:", list(input_data.columns))
            missing = set(model.feature_names_in_) - set(input_data.columns)
            extra = set(input_data.columns) - set(model.feature_names_in_)
            if missing:
                st.error(f"Missing columns: {missing}")
            if extra:
                st.warning(f"Extra columns: {extra}")
