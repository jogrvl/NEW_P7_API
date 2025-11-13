# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Seuil métier optimal
THRESHOLD_METIER = 0.54

# Charger le pipeline de prédiction
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "modele_pipeline.pkl")
pipe = joblib.load(MODEL_PATH)

# Charger les données clients
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "train_df_cleaned.csv")
df_clients = pd.read_csv(DATA_PATH)

# Vérification basique
if "SK_ID_CURR" not in df_clients.columns:
    raise ValueError("La colonne 'SK_ID_CURR' est absente du fichier train_df_cleaned.csv")

# Création de l'application FastAPI
app = FastAPI(
    title="API Scoring Crédit P7",
    version="2.0",
    description="API de scoring crédit basée sur un modèle LightGBM, entrée = SK_ID_CURR uniquement."
)

# --- Schéma d’entrée : uniquement le numéro client ---
class ClientID(BaseModel):
    SK_ID_CURR: int

@app.get("/")
def root():
    return {"message": "✅ API Scoring Crédit - opérationnelle."}

@app.post("/predict")
def predict_by_client_id(data: ClientID):
    client_id = data.SK_ID_CURR

    # Vérifier que le client existe
    if client_id not in df_clients["SK_ID_CURR"].values:
        raise HTTPException(status_code=404, detail=f"Client {client_id} introuvable dans la base.")

    # Extraire les features du client
    client_data = df_clients[df_clients["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"])

    # Vérifier cohérence colonnes modèle
    model_columns = pipe.feature_names_in_
    missing_cols = set(model_columns) - set(client_data.columns)
    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Colonnes manquantes dans les données client : {missing_cols}"
        )

    # Réordonner les colonnes pour correspondre au modèle
    client_data = client_data[model_columns]

    # Prédiction
    proba = float(pipe.predict_proba(client_data)[0][1])
    prediction = int(proba > THRESHOLD_METIER)

    return {
        "SK_ID_CURR": client_id,
        "score_probabilite": round(proba, 4),
        "prediction": "Refusé" if prediction == 1 else "Approuvé",
        "seuil_utilise": THRESHOLD_METIER
    }
