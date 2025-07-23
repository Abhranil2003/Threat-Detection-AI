from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from utils import load_artifacts, preprocess_input, predict_input

app = FastAPI(title="AI Threat Detection API")

# Load model artifacts at startup
MODEL_PATH = "models/model_latest.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
ENCODERS_PATH = "data/processed/encoders.pkl"

try:
    model, scaler, encoders = load_artifacts(MODEL_PATH, SCALER_PATH, ENCODERS_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# Example input model (extend fields as per your features)
class TrafficSample(BaseModel):
    Protocol: str
    Flow_Duration: int
    Total_Fwd_Packets: int
    Total_Backward_Packets: int
    # Add all necessary features here

@app.post("/predict")
def predict(sample: TrafficSample):
    try:
        input_dict = sample.dict()
        df = pd.DataFrame([input_dict])
        X = preprocess_input(df, encoders, scaler)
        prediction = predict_input(model, X)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to Threat Detection API v1"}