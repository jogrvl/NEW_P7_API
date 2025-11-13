from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict():
    payload = {"SK_ID_CURR": 100001}

    response = client.post("/predict", json=payload)

    # Vérification que la route répond
    assert response.status_code in [200, 404]

    # Si le client existe, on teste le contenu
    if response.status_code == 200:
        data = response.json()
        assert "score_probabilite" in data
        assert "prediction" in data
