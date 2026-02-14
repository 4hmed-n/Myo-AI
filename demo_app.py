import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('myocore_pipeline.pkl')

st.title('Myo-Sim Bio-Deck — Chronos Time-Travel Interface')

# Feature order (update to match your model's training columns)
feature_names = ['age', 'sex', 'trestbps', 'chol', 'smoke', 'weight', 'height']

# --- Professional Layout ---
st.markdown("""
<style>
.big-font {font-size:28px !important; font-weight:bold; color:#2c3e50;}
.section-title {font-size:20px !important; font-weight:bold; color:#2980b9; margin-top: 1em;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,2])

with col1:
    st.markdown('<div class="section-title">Patient Vitals</div>', unsafe_allow_html=True)
    age = st.slider('Age (years in future)', 0, 20, 0)
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    trestbps = st.slider('Systolic Blood Pressure (mmHg)', 80, 200, 120)
    chol = st.slider('Cholesterol (mg/dL)', 100, 600, 200)
    smoke = st.selectbox('Smoker', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    weight = st.slider('Weight (kg)', 30, 200, 75)
    height = st.slider('Height (cm)', 100, 220, 170)
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    predict_btn = st.button('Predict', use_container_width=True)

with col2:
    st.markdown('<div class="big-font">Chronos Time-Travel Risk Dashboard</div>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1511174511562-5f97f2b2e2b9?auto=format&fit=crop&w=800&q=80", use_column_width=True)

patient = {
    'age': age,
    'sex': sex,
    'trestbps': trestbps,
    'chol': chol,
    'smoke': smoke,
    'weight': weight,
    'height': height
}

input_df = pd.DataFrame([patient], columns=feature_names)

if predict_btn:
    prob = model.predict_proba(input_df)[0, 1] if hasattr(model, 'predict_proba') else model.predict(input_df)[0]
    status = 'HIGH RISK' if prob > 0.5 else 'LOW RISK'
    status_color = '#e74c3c' if prob > 0.5 else '#2ecc71'
    st.markdown(f"### CVD Probability: <span style='color:{status_color}'>{prob:.1%}</span> — <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    st.write(f"Simulated Age: {age}")
    # Calculate BMI and Pulse Pressure using correct variables
    bmi = weight / ((height / 100) ** 2) if height > 0 else 0
    pulse_pressure = trestbps - dia_bp

    # Output section (right column)
    with col2:
        st.markdown(f"### <span style='color:{status_color}'>CVD Probability: {prob:.1%} — {status}</span>", unsafe_allow_html=True)
        st.write(f"Simulated Age: {age}")
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

# --- Thicken sliders with custom CSS ---
st.markdown("""
<style>
[data-testid="stSlider"] .st-c2 {
    height: 0.7rem;
}
[data-testid="stSlider"] .st-c1 {
    height: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

