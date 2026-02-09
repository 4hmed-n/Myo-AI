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

# --- Clinical Features ---
age = st.sidebar.slider("Age (Years)", 20, 100, 50)
gender_opt = st.sidebar.radio("Sex", ["Male", "Female"])
sex = 1 if gender_opt == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], index=0)
trestbps = st.sidebar.slider("Resting BP (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)

fbs_opt = st.sidebar.radio("Fasting BS > 120 mg/dl?", ["No", "Yes"])
fbs = 1 if fbs_opt == "Yes" else 0

restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2], index=0)
thalachh = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

exng_opt = st.sidebar.radio("Exercise Induced Angina?", ["No", "Yes"])
exng = 1 if exng_opt == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slp = st.sidebar.selectbox("Slope", [0, 1, 2], index=1)
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

# 4. Run Simulation
if st.button("Run Simulation"):
    
    # Calculate derived features that the model expects
    pulse_pressure = trestbps - 80  # Systolic - Diastolic (using 80 as default diastolic)
    sensor_signal_available = 1     # Default: signal is available
    
    # Build input DataFrame with EXACT column names the model expects
    # Based on your error: model expects 'ca', not 'caa', and 'exang', not 'exng'
    input_data = pd.DataFrame({
        # Core features (using standard UCI Cleveland naming)
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalachh],      # Note: model expects 'thalach', not 'thalachh'
        'exang': [exng],            # Changed from 'exng' to 'exang'
        'oldpeak': [oldpeak],
        'slope': [slp],             # Changed from 'slp' to 'slope'
        'ca': [caa],                # Changed from 'caa' to 'ca'
        'thal': [thall],            # Changed from 'thall' to 'thal'
        
        # Additional features the model expects (from your error message)
        'pulse_pressure': [pulse_pressure],
        'sensor_signal_available': [sensor_signal_available],
        
        # Lifestyle features
        'bmi': [bmi],
        'smoke': [1 if smoke else 0],
        'alco': [1 if alco else 0],
        'active': [1 if active else 0],
    })

    # DEBUG: Show what columns the model actually expects (uncomment if needed)
    # if hasattr(model, 'feature_names_in_'):
    #     st.write("Model expects:", list(model.feature_names_in_))
    #     st.write("You provided:", list(input_data.columns))
    #     st.write("Missing:", set(model.feature_names_in_) - set(input_data.columns))
    #     st.write("Extra:", set(input_data.columns) - set(model.feature_names_in_))

    try:
        # Ensure column order matches training data
        if hasattr(model, 'feature_names_in_'):
            # Reorder columns to match exactly what model was trained on
            # Only keep columns that exist in both
            expected_cols = list(model.feature_names_in_)
            available_cols = [col for col in expected_cols if col in input_data.columns]
            input_data = input_data[available_cols]
        
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
        # Debug info
        if hasattr(model, 'feature_names_in_'):
            st.write("Model expects:", list(model.feature_names_in_))
            st.write("You provided:", list(input_data.columns))
            missing = set(model.feature_names_in_) - set(input_data.columns)
            extra = set(input_data.columns) - set(model.feature_names_in_)
            if missing:
                st.error(f"Missing columns: {missing}")
            if extra:
                st.warning(f"Extra columns (will be ignored): {extra}")
