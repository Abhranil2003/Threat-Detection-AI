import os
import numpy as np
import joblib
import json
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup directories
LOGS_DIR = os.path.join("reports", "logs")
METRICS_DIR = os.path.join("reports", "metrics")
FIGURES_DIR = os.path.join("reports", "figures")
REPORTS_DIR = os.path.join("reports", "model_reports")

# Create necessary directories
for directory in [LOGS_DIR, METRICS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "evaluation.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_test_data(processed_path):
    try:
        X_test = np.load(os.path.join(processed_path, "X_test.npy"))
        y_test = np.load(os.path.join(processed_path, "y_test.npy"))
        return X_test, y_test
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        return None, None

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def save_metrics(metrics, filename):
    with open(os.path.join(METRICS_DIR, filename), 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved metrics to {filename}")

def save_classification_report(report, filename):
    with open(os.path.join(REPORTS_DIR, filename), 'w') as f:
        f.write(report)
    logging.info(f"Saved classification report to {filename}")

def plot_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()
    logging.info(f"Saved confusion matrix to {filename}")

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1_score": f1_score(y_test, y_pred, average='macro')
    }
    save_metrics(metrics, "evaluation_metrics.json")

    # Classification report
    report = classification_report(y_test, y_pred)
    save_classification_report(report, "classification_report.txt")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test)))
    plot_confusion_matrix(cm, labels, "confusion_matrix.png")

    logging.info("Evaluation complete.")

# Example usage
if __name__ == "__main__":
    processed_path = "data/processed"
    model_path = "models/model_latest.pkl"

    model = load_model(model_path)
    X_test, y_test = load_test_data(processed_path)

    if model and X_test is not None:
        evaluate(model, X_test, y_test)
