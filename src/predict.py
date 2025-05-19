import os
import numpy as np
import pandas as pd
import joblib

def load_artifacts(model_path, scaler_path, encoders_path):
    """Load model, scaler, and encoders."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    return model, scaler, encoders

def preprocess_input(df, encoders, scaler):
    """Encode and scale input features."""
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    
    X = scaler.transform(df)
    return X

def predict_input(model, X):
    """Run prediction."""
    return model.predict(X)

# Example usage
if __name__ == "__main__":
    model_path = "models/model_latest.pkl"
    scaler_path = "data/processed/scaler.pkl"
    encoders_path = "data/processed/encoders.pkl"

    # Simulated single sample input
    input_data = {
        "Protocol": ["TCP"],
        "Flow Duration": [123456],
        "Total Fwd Packets": [10],
        "Total Backward Packets": [5],
        # ... include all required columns
    }

    df = pd.DataFrame(input_data)

    model, scaler, encoders = load_artifacts(model_path, scaler_path, encoders_path)
    X_processed = preprocess_input(df, encoders, scaler)
    prediction = predict_input(model, X_processed)

    print(f"[INFO] Prediction: {prediction[0]}")
