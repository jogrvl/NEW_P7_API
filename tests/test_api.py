# tests/test_api.py
import requests

API_URL = "https://new-p7-api.onrender.com/predict"


def test_predict_endpoint():
    """Test basique : l’API répond correctement"""

    payload = {"SK_ID_CURR": 134561}

    response = requests.post(API_URL, json=payload)

    # Vérifie que la requête HTTP est bien acceptée
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "client_id" in data
        assert "score_probabilite" in data
        assert "decision" in data

    if response.status_code == 404:
        assert "detail" in response.json()
