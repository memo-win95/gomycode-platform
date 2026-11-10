# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_risk_valid_payload():
    payload = {
        "attendance_rate": 0.90,
        "assignments_completed_ratio": 0.85,
        "platform_time_hours": 45.0
    }
    response = client.post("/predict-dropout-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "dropout_risk_score" in data
    assert data["status"] in ["Normal", "High Risk"]

def test_predict_risk_invalid_data_rejection():
    # Envoi d'un taux de présence invalide (> 1.0)
    invalid_payload = {
        "attendance_rate": 1.5,
        "assignments_completed_ratio": 0.5,
        "platform_time_hours": 10.0
    }
    response = client.post("/predict-dropout-risk", json=invalid_payload)
    assert response.status_code == 422  # Erreur de validation Pydantic