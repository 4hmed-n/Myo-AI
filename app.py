import streamlit as st
import pandas as pd
import joblib

# 1. Page Setup
st.set_page_config(page_title="Myo AI Simulator", layout="centered")
st.title("🫀 Myo AI: Cardiovascular Risk Simulator")
st.caption("Powered by Myo-Core Engine (Random Forest + Gradient Boosting)")

# 2. Load the Model
try:
    # This looks for the file you just renamed
    model = joblib.load('myocore_pipeline.pkl')
    st.success("System Online: Neural Link Established")
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please upload 'myocore_pipeline.pkl' to GitHub.")
    st.stop()

# 3. Sidebar Inputs (Patient Vitals)
st.sidebar.header("Patient Vitals")

# We use standard ranges for these inputs
age = st.sidebar.slider("Age", 20, 100, 50)
gender = st.sidebar.radio("Sex", ["Male", "Female"])
sex = 1 if gender == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
ap_hi = st.sidebar.slider("Systolic Blood Pressure (ap_hi)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 250)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
fbs_val = 1 if fbs == "Yes" else 0

restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalachh = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exng = st.sidebar.radio("Exercise Induced Angina?", ["No", "Yes"])
exng_val = 1 if exng == "Yes" else 0

oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slp = st.sidebar.selectbox("Slope", [0, 1, 2])
caa = st.sidebar.slider("Major Vessels (0-4)", 0, 4, 0)
thall = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

# 4. Prediction Button
if st.button("Run Simulation"):
    # Create a DataFrame with the EXACT column names your model expects
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trtbps': [ap_hi], # Mapping ap_hi to trtbps if your model uses that name
        'chol': [chol],
        'fbs': [fbs_val],
        'restecg': [restecg],
        'thalachh': [thalachh],
        'exng': [exng_val],
        'oldpeak': [oldpeak],
        'slp': [slp],
        'caa': [caa],
        'thall': [thall]
    })

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
        
        with col2:
            if probability > 0.5:
                st.error("⚠️ HIGH RISK DETECTED")
            else:
                st.success("✅ LOW RISK DETECTED")
                
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Check if your column names in app.py match your notebook exactly.")