from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Document Classification API is running"}

def test_predict():
    test_text = "esquire radio and electronics inc ee th qtr shr profit"
    response = client.post(
        "/predict",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert isinstance(data["predicted_class"], int)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["probabilities"], list)

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422  # Validation error 