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

# 3. Sidebar Inputs
st.sidebar.header("Patient Vitals")

# --- Clinical Features ---
age = st.sidebar.slider("Age (Years)", 20, 100, 50)
gender_opt = st.sidebar.radio("Sex", ["Male", "Female"])
sex_val = 1 if gender_opt == "Male" else 0

# Match both naming conventions (sex vs gender) just in case
gender_val = 2 if gender_opt == "Male" else 1 

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)
trestbps = st.sidebar.slider("Resting BP (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)

fbs_opt = st.sidebar.radio("Fasting BS > 120 mg/dl?", ["No", "Yes"])
fbs_val = 1 if fbs_opt == "Yes" else 0

restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2], index=0)
thalachh = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

exng_opt = st.sidebar.radio("Exercise Induced Angina?", ["No", "Yes"])
exng_val = 1 if exng_opt == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slp = st.sidebar.selectbox("Slope", [0, 1, 2], index=1)
caa = st.sidebar.slider("Major Vessels (0-4)", 0, 4, 0)
thall = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3], index=2)

# --- Lifestyle Features (For the Fusion part) ---
st.sidebar.markdown("---")
st.sidebar.header("Lifestyle & Physical")
height = st.sidebar.slider("Height (cm)", 100, 220, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 75)

# Calculate BMI automatically
bmi = weight / ((height/100)**2)

smoke = st.sidebar.checkbox("Smoker?")
alco = st.sidebar.checkbox("Alcohol Intake?")
active = st.sidebar.checkbox("Physically Active?")

# --- ECG Signal Features (HIDDEN DEFAULTS) ---
# The model needs these, but users can't type them. We use "Normal" defaults.
# (You can make these sliders if you really want, but defaults are safer for a demo)
ecg_mean = 0.0  
ecg_std = 1.0   
ecg_skew = 0.0  
ecg_kurtosis = 3.0 # Normal distribution kurtosis

# 4. Run Simulation
if st.button("Run Simulation"):
    # We construct a DataFrame with EVERY POSSIBLE COLUMN the model might want.
    # We include duplicates (e.g. 'fbs' and 'fastingbs') to catch all naming variations.
    input_data = pd.DataFrame({
        # Dataset 1 Features (Heart Disease)
        'age': [age],
        'sex': [sex_val],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_val],
        'restecg': [restecg],
        'thalachh': [thalachh],
        'exng': [exng_val],
        'oldpeak': [oldpeak],
        'slp': [slp],
        'caa': [caa],
        'thall': [thall],
        
        # Dataset 2 Features (Cardio Disease)
        'gender': [gender_val],
        'height': [height],
        'weight': [weight],
        'ap_hi': [trestbps], # Map Systolic BP here
        'ap_lo': [80],       # Default Diastolic if not asked
        'cholesterol': [1 if chol < 200 else 2], # Map numeric chol to categorical
        'gluc': [1 if fbs_val == 0 else 2],      # Map fbs to gluc
        'smoke': [1 if smoke else 0],
        'alco': [1 if alco else 0],
        'active': [1 if active else 0],
        'bmi': [bmi],

        # Dataset 3 Features (The missing ones from your error!)
        'fastingbs': [fbs_val],   # The error specifically asked for this name
        'restingbp': [trestbps],  # Another common name for BP
        'maxhr': [thalachh],      # Another common name for Heart Rate
        
        # ECG Signal Features (The other missing ones!)
        'ecg_mean': [ecg_mean],
        'ecg_std': [ecg_std],
        'ecg_skew': [ecg_skew],
        'ecg_kurtosis': [ecg_kurtosis]
    })

    try:
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
            st.caption(f"Patient BMI: {bmi:.1f}")
        
        with col2:
            if probability > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
            else:
                st.success("✅ LOW RISK DETECTED")
                
    except Exception as e:
        st.error(f"Error: {e}")
        # This will print the EXACT list of columns the model is still missing (if any)
        if hasattr(model, 'feature_names_in_'):
            st.write("Columns still missing:", set(model.feature_names_in_) - set(input_data.columns))
