import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('myocore_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('ECG Patient Outcome Predictor')

# Example slider for a single feature (customize as needed)
feature = st.slider('ECG Feature Value', 0.0, 1.0, 0.5)

# Prepare input for prediction (adjust shape as needed)
input_data = np.array([[feature]])

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Prediction: {prediction[0]}')
