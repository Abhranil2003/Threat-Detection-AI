import os
import argparse
import numpy as np
from sklearn.model_selection import cross_val_score
from datetime import datetime
from models.model_factory import get_model
import joblib
import json
import logging

# Directories for reports
REPORTS_DIR = os.path.join("reports", "metrics")
LOGS_DIR = os.path.join("reports", "logs")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
log_file = os.path.join(LOGS_DIR, "training.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(processed_path):
    try:
        X_train = np.load(os.path.join(processed_path, "X_train.npy"))
        y_train = np.load(os.path.join(processed_path, "y_train.npy"))
        return X_train, y_train
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        return None, None

def train_model(model_name, X_train, y_train):
    model = get_model(model_name)
    model.train(X_train, y_train)
    logging.info("Training complete.")
    return model

def evaluate_model(model, X_train, y_train, model_name):
    try:
        scores = cross_val_score(model.model, X_train, y_train, cv=5, scoring='f1_weighted')
        mean_score = scores.mean()
        std_score = scores.std()
        logging.info(f"Cross-validated F1-score: {mean_score:.4f} ± {std_score:.4f}")

        # Save metrics
        metrics = {
            "model": model_name,
            "f1_score_mean": mean_score,
            "f1_score_std": std_score,
            "timestamp": datetime.now().isoformat()
        }
        metrics_file = os.path.join(REPORTS_DIR, f"train_metrics_{model_name}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_file}")

    except Exception as e:
        logging.error(f"Cross-validation failed: {e}")

def save_model(model, output_path, model_name):
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, f"{model_name}_{timestamp}.pkl")
    model.save(model_path)
    logging.info(f"Model saved to: {model_path}")

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
