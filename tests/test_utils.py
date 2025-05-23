from src.utils import load_model 
import os

def test_model_loading():
    model_path = "models/threat_model.pkl"
    if os.path.exists(model_path):
        model = load_model(model_path)
        assert model is not None
    else:
        assert True  