import os
import argparse
import numpy as np
from datetime import datetime
from models.model_factory import get_model
from evaluate import evaluate_model
import joblib

def load_data(processed_path):
    try:
        X_train = np.load(os.path.join(processed_path, "X_train.npy"))
        y_train = np.load(os.path.join(processed_path, "y_train.npy"))
        return X_train, y_train
    except Exception as e:
        print(f"[ERROR] Failed to load training data: {e}")
        return None, None

def train_model(model_name, X_train, y_train):
    model = get_model(model_name)
    model.train(X_train, y_train)
    print("[INFO] Training complete.")
    return model

def save_model(model, output_path, model_name):
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, f"{model_name}_{timestamp}.pkl")
    model.save(model_path)
    print(f"[INFO] Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a threat detection model")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "logistic_regression", "xgboost"],
                        help="Specify the model to train")
    args = parser.parse_args()

    processed_data_path = "data/processed"
    model_output_path = "models"

    X_train, y_train = load_data(processed_data_path)

    if X_train is not None and y_train is not None:
        model = train_model(args.model, X_train, y_train)
        evaluate_model(model, X_train, y_train, args.model)
        save_model(model, model_output_path, args.model)
