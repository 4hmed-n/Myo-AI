import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('myocore_pipeline.pkl')

st.title('Myo-Sim Bio-Deck — Chronos Time-Travel Interface')

# Feature input widgets
age = st.slider('Age', 18, 100, 45)
sys_bp = st.slider('Systolic BP', 80, 220, 130)
dia_bp = st.slider('Diastolic BP', 40, 130, 80)
cholesterol = st.slider('Cholesterol', 50, 600, 200, step=5)
weight = st.slider('Weight (kg)', 30, 200, 75)
height = st.slider('Height (cm)', 100, 220, 170)
smoker = st.checkbox('Smoker', value=False)
active = st.checkbox('Active', value=True)
years_ahead = st.slider('⏳ Years Ahead', 0, 20, 0)

# Derived features
sim_age = age + years_ahead
bmi = weight / ((height / 100) ** 2) if height > 0 else 0
pulse_pressure = sys_bp - dia_bp

# Feature order (update to match your model's training columns)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Streamlit widgets for each feature
age = st.slider('Age', 18, 100, 45)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.slider('Chest Pain Type (cp)', 0, 3, 0)
trestbps = st.slider('Resting Blood Pressure (trestbps)', 80, 200, 120)
chol = st.slider('Cholesterol (chol)', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
restecg = st.slider('Resting ECG (restecg)', 0, 2, 1)
thalach = st.slider('Max Heart Rate (thalach)', 60, 220, 150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.0, 1.0, step=0.1)
slope = st.slider('Slope of ST (slope)', 0, 2, 1)
ca = st.slider('Number of Major Vessels (ca)', 0, 4, 0)
thal = st.slider('Thalassemia (thal)', 0, 3, 1)

patient = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

input_df = pd.DataFrame([patient], columns=feature_names)

if st.button('Predict'):
    prob = model.predict_proba(input_df)[0, 1] if hasattr(model, 'predict_proba') else model.predict(input_df)[0]
    status = 'HIGH RISK' if prob > 0.5 else 'LOW RISK'
    status_color = '#e74c3c' if prob > 0.5 else '#2ecc71'
    st.markdown(f"### CVD Probability: <span style='color:{status_color}'>{prob:.1%}</span> — <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    st.write(f"Simulated Age: {sim_age}")
    st.write(f"BMI: {bmi:.1f}")
    st.write(f"Pulse Pressure: {pulse_pressure} mmHg")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "CVD Risk Gauge"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': status_color},
            'steps': [
                {'range': [0, 40], 'color': '#2ecc71'},
                {'range': [40, 70], 'color': '#f39c12'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Chronos projection
    years = list(range(0, 21))
    risks = []
    for y in years:
        patient_proj = patient.copy()
        patient_proj['age'] = age + y
        input_proj = pd.DataFrame([patient_proj], columns=feature_names)
        p = model.predict_proba(input_proj)[0, 1] if hasattr(model, 'predict_proba') else model.predict(input_proj)[0]
        risks.append(p)
    ages = [age + y for y in years]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ages, y=risks, mode='lines+markers', name='Projected CVD Risk', line=dict(color='#e74c3c')))
    fig2.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Risk Threshold (50%)", annotation_position="top left")
    fig2.update_layout(title="Chronos Engine: 20-Year Risk Projection", xaxis_title="Age (years)", yaxis_title="CVD Probability", yaxis_range=[0,1])
    st.plotly_chart(fig2, use_container_width=True)
