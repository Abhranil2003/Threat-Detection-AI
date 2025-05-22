def get_model(model_name):
    if model_name == "random_forest":
        from models.random_forest import RandomForestModel
        return RandomForestModel()
    elif model_name == "logistic_regression":
        from models.logistic_regression import LogisticRegressionModel
        return LogisticRegressionModel()
    elif model_name == "xgboost":
        from models.xgboost_model import XGBoostModel
        return XGBoostModel()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")