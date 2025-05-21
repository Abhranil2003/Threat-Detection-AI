from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/v1/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Threat Detection API v1"}

def test_prediction():
    sample_data = {
        "TotLen Fwd Pkts": 20,
        "Protocol": "TCP",
        "Src IP": "192.168.1.1",
        "Dst IP": "192.168.1.2"
    }
    response = client.post("/v1/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()