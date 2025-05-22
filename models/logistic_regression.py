from sklearn.linear_model import LogisticRegression
import joblib
from models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)