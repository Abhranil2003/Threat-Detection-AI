from xgboost import XGBClassifier
import joblib

class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
