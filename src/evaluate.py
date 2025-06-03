import os
import numpy as np
from sklearn.model_selection import cross_val_score
from datetime import datetime
import json

from models.model_factory import get_model

def evaluate_model(model, X_train, y_train, model_name):
    try:
        scores = cross_val_score(model.model, X_train, y_train, cv=5, scoring='f1_weighted')
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"[INFO] Cross-validated F1-score: {mean_score:.4f} ± {std_score:.4f}")

        metrics = {
            "model": model_name,
            "f1_score_mean": mean_score,
            "f1_score_std": std_score,
            "timestamp": datetime.now().isoformat()
        }

        os.makedirs("reports/metrics", exist_ok=True)
        with open(f"reports/metrics/train_metrics_{model_name}.json", "w") as f:
            json.dump(metrics, f, indent=4)

    except Exception as e:
        print(f"[ERROR] Cross-validation failed: {e}")
