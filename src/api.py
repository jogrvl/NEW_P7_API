# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# -----------------------------
# Paramètres
# -----------------------------
THRESHOLD_METIER = 0.54
BASE_DIR = os.path.dirname(__file__)

# Pipeline
MODEL_PATH = os.path.join(BASE_DIR, "..", "modele_pipeline.pkl")

# Dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "train_df_sample.csv")

# -----------------------------
# Chargement modèle + données
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ modele_pipeline.pkl introuvable à la racine du projet.")

pipe = joblib.load(MODEL_PATH)

df_clients = pd.read_csv(DATA_PATH)
if "SK_ID_CURR" not in df_clients.columns:
    raise KeyError("❌ La colonne 'SK_ID_CURR' est manquante dans le dataset chargé.")
df_clients.set_index("SK_ID_CURR", inplace=True)

ALL_COLUMNS = pipe.feature_names_in_

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="API Scoring Crédit P7", version="1.0")

class ClientRequest(BaseModel):
    SK_ID_CURR: int

@app.get("/")
def root():
    return {"message": "API Scoring Crédit - OK"}

@app.get("/clients")
def get_clients():
    """
    Retourne la liste des SK_ID_CURR disponibles
    """
    return df_clients.index.tolist()

@app.post("/predict")
def predict(request: ClientRequest):
    client_id = request.SK_ID_CURR

    if client_id not in df_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")

    client_data = df_clients.loc[client_id].to_dict()

    # Crée l'input complet avec toutes les colonnes attendues par le pipeline
    full_input = {col: 0.0 for col in ALL_COLUMNS}
    for col in client_data:
        if col in ALL_COLUMNS:
            full_input[col] = client_data[col]

    df_input = pd.DataFrame([full_input])

    proba = float(pipe.predict_proba(df_input)[0][1])
    decision = int(proba > THRESHOLD_METIER)

    return {
        "client_id": client_id,
        "score_probabilite": round(proba, 4),
        "decision": "Refusé" if decision == 1 else "Approuvé",
        "seuil": THRESHOLD_METIER
    }
