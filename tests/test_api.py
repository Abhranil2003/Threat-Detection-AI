from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Threat Detection API v1"}

def test_prediction():
    sample_data = {
        "Protocol": "TCP",
        "Flow_Duration": 100,
        "Total_Fwd_Packets": 10,
        "Total_Backward_Packets": 5
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()