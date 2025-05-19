import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

def load_data(processed_path):
    """Load preprocessed training data."""
    try:
        X_train = np.load(os.path.join(processed_path, "X_train.npy"))
        y_train = np.load(os.path.join(processed_path, "y_train.npy"))
        return X_train, y_train
    except Exception as e:
        print(f"[ERROR] Failed to load training data: {e}")
        return None, None

def train_model(X_train, y_train, n_estimators=100):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    print("[INFO] Training complete.")
    return model

def evaluate_model(model, X_train, y_train):
    """Evaluate model using cross-validation and print metrics."""
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f"[INFO] Cross-validated F1-score: {scores.mean():.4f} ± {scores.std():.4f}")

def save_model(model, output_path):
    """Save trained model to disk."""
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, f"model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")

# Example usage
if __name__ == "__main__":
    processed_data_path = "data/processed"
    model_output_path = "models"

    X_train, y_train = load_data(processed_data_path)
    
    if X_train is not None and y_train is not None:
        model = train_model(X_train, y_train)
        evaluate_model(model, X_train, y_train)
        save_model(model, model_output_path)
