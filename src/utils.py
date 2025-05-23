import joblib
import pandas as pd
import pickle

def load_artifacts(model_path, scaler_path, encoders_path):
    """Load model, scaler, and encoders."""
    if not all(map(os.path.exists, [model_path, scaler_path, encoders_path])):
        raise FileNotFoundError("One or more artifacts missing.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    return model, scaler, encoders

def preprocess_input(df, encoders, scaler):
    """Apply encoding and scaling to new data."""
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    X = scaler.transform(df)
    return X

def predict_input(model, X):
    """Predict using trained model."""
    return model.predict(X)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model