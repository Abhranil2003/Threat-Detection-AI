import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data(processed_path):
    """Load preprocessed test data."""
    try:
        X_test = np.load(os.path.join(processed_path, "X_test.npy"))
        y_test = np.load(os.path.join(processed_path, "y_test.npy"))
        return X_test, y_test
    except Exception as e:
        print(f"[ERROR] Failed to load test data: {e}")
        return None, None

def load_model(model_path):
    """Load the trained model."""
    try:
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def evaluate(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    print("[INFO] Confusion matrix saved to reports/confusion_matrix.png")

# Example usage
if __name__ == "__main__":
    processed_path = "data/processed"
    model_path = "models/model_latest.pkl"  # or update with latest timestamp

    model = load_model(model_path)
    X_test, y_test = load_test_data(processed_path)

    if model and X_test is not None:
        evaluate(model, X_test, y_test)
