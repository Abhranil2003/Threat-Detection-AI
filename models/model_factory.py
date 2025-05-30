from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel

def get_model(model_name):
    model_name = model_name.lower()
    if model_name == "random_forest":
        return RandomForestModel()
    elif model_name == "logistic_regression":
        return LogisticRegressionModel()
    elif model_name == "xgboost":
        return XGBoostModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
