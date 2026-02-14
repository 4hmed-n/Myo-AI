import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('myocore_pipeline.pkl')

st.title('ECG Patient Outcome Predictor')

# Example slider for a single feature (customize as needed)
feature = st.slider('ECG Feature Value', 0.0, 1.0, 0.5)

# Prepare input for prediction (adjust shape as needed)
input_data = np.array([[feature]])

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Prediction: {prediction[0]}')
