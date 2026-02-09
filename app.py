import gradio as gr
import pandas as pd
import joblib
import numpy as np

# 1. Load the Model
try:
    model = joblib.load('myocore_pipeline.pkl')
except:
    model = None

def predict_risk(age, sex, height, weight, ap_hi, ap_lo, chol_val, gluc_val, smoke, alco, active):
    if model is None:
        return "Model not found. Please upload 'myocore_pipeline.pkl'"
    
    # Calculate Engineered Features
    bmi = weight / ((height/100)**2)
    pulse_pressure = ap_hi - ap_lo
    
    # Map inputs to match your Fusion Model's expected columns
    # Using the exact order your model requires
    columns = [
        'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak', 'heartdisease',
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'ecg_mean', 'ecg_std', 'ecg_skew', 'ecg_kurtosis',
        'sensor_signal_available', 'bmi', 'pulse_pressure'
    ]
    
    # Create the data row with your defaults
    gender = 2 if sex == "Male" else 1
    chol_cat = 1 if chol_val == "Normal" else (2 if chol_val == "Above Normal" else 3)
    gluc_cat = 1 if gluc_val == "Normal" else 2

    row = [
        age, ap_hi, chol_cat, 0, 150, 1.0, 0, # Clinical set
        age, gender, height, weight, ap_hi, ap_lo, chol_cat, gluc_cat, # Cardio set
        int(smoke), int(alco), int(active), 0.0, 1.0, 0.0, 3.0, # ECG & Lifestyle
        1, bmi, pulse_pressure # Sensor & Engineered
    ]
    
    input_df = pd.DataFrame([row], columns=columns)
    
    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk_level = "⚠️ HIGH RISK" if prob > 0.5 else "✅ LOW RISK"
    
    return f"Prediction: {risk_level}\nProbability: {prob*100:.1f}%\nBMI: {bmi:.1f}"

# 2. Build the Interface
interface = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(20, 100, value=50, label="Age"),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Slider(100, 220, value=170, label="Height (cm)"),
        gr.Slider(30, 150, value=75, label="Weight (kg)"),
        gr.Slider(90, 200, value=120, label="Systolic BP"),
        gr.Slider(50, 130, value=80, label="Diastolic BP"),
        gr.Dropdown(["Normal", "Above Normal", "Well Above Normal"], label="Cholesterol"),
        gr.Dropdown(["Normal", "Above Normal"], label="Glucose"),
        gr.Checkbox(label="Smoker"),
        gr.Checkbox(label="Alcohol Intake"),
        gr.Checkbox(label="Physically Active")
    ],
    outputs="text",
    title="🫀 Myo AI: Cardiovascular Risk Simulator",
    description="Fusion Model Prediction Engine"
)

interface.launch()
