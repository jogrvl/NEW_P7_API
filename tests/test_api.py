def test_predict():
    payload = {
        "SK_ID_CURR": 100001  # Remplace par un ID client existant dans train_df_cleaned.csv
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert "score_probabilite" in json_resp
    assert "prediction" in json_resp
    assert "seuil_utilise" in json_resp
