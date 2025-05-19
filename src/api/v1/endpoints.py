from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd

from utils import load_artifacts, preprocess_input, predict_input

router = APIRouter()

# Load model artifacts once
model, scaler, encoders = load_artifacts(
    "models/model_latest.pkl",
    "data/processed/scaler.pkl",
    "data/processed/encoders.pkl"
)

class TrafficSample(BaseModel):
    Protocol: str
    Flow_Duration: int
    Total_Fwd_Packets: int
    Total_Backward_Packets: int
    # Add all required fields

@router.post("/predict")
def predict(sample: TrafficSample):
    try:
        input_df = pd.DataFrame([sample.dict()])
        X = preprocess_input(input_df, encoders, scaler)
        pred = predict_input(model, X)
        return {"prediction": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
