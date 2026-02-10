import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
try:
    pipeline = joblib.load("myocore_pipeline.pkl")
    print("✅ Model Loaded!")
except:
    print("❌ Error: myocore_pipeline.pkl not found")
    pipeline = None

# Input Schema (Matches your Model)
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int

@app.post("/predict")
def predict(data: PatientData):
    if not pipeline: raise HTTPException(500, "Model missing")
    df = pd.DataFrame([data.dict()])
    prob = pipeline.predict_proba(df)[0][1]
    return {"risk": "High" if prob > 0.5 else "Low", "probability": float(prob)}

# --- 🆕 NEW: CHRONOS TIME TRAVEL ENGINE ---
@app.post("/simulate")
def simulate(data: PatientData):
    """
    Predicts risk for the next 20 years by incrementing age.
    """
    if not pipeline: raise HTTPException(500, "Model missing")
    
    results = []
    base_data = data.dict()
    start_age = int(data.age)

    # Loop from Now (0) to +20 years
    for year_offset in range(21):
        # Create a hypothetical future patient
        future_patient = base_data.copy()
        future_patient['age'] = start_age + year_offset
        
        # Predict their risk
        df = pd.DataFrame([future_patient])
        prob = pipeline.predict_proba(df)[0][1]
        
        results.append({
            "year_offset": year_offset,
            "age": start_age + year_offset,
            "probability": float(prob)
        })
        
    return results