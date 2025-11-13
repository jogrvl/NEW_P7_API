# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import requests

# -----------------------------
# Paramètres
# -----------------------------
THRESHOLD_METIER = 0.54

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "modele_pipeline.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "train_df_cleaned.csv")
DRIVE_DOWNLOAD_URL = (
    "https://drive.google.com/uc?export=download&id=1pOgCnUZYEmmhevjbj2jBQU2Tre0uhUG6"
)

# -----------------------------
# Télécharger automatiquement le CSV si absent
# -----------------------------
def download_csv_if_needed():
    if os.path.exists(DATA_PATH):
        return

    print("➡ Téléchargement du dataset depuis Google Drive...")
    response = requests.get(DRIVE_DOWNLOAD_URL, stream=True)
    response.raise_for_status()

    with open(DATA_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✔ train_df_cleaned.csv téléchargé")


download_csv_if_needed()

# -----------------------------
# Chargement pipeline + données
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ modele_pipeline.pkl introuvable à la racine du projet.")

pipe = joblib.load(MODEL_PATH)

df_clients = pd.read_csv(DATA_PATH)

if "SK_ID_CURR" not in df_clients.columns:
    raise KeyError("❌ La colonne 'SK_ID_CURR' est manquante dans train_df_cleaned.csv")

df_clients.set_index("SK_ID_CURR", inplace=True)

# Colonnes utilisées par le modèle
ALL_COLUMNS = pipe.feature_names_in_

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="API Scoring Crédit P7", version="1.0")

class ClientRequest(BaseModel):
    SK_ID_CURR: int

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "API Scoring Crédit - OK"}

@app.post("/predict")
def predict(request: ClientRequest):

    client_id = request.SK_ID_CURR

    # Vérifier si le client est connu
    if client_id not in df_clients.index:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} non trouvé dans la base."
        )

    # Récupérer les données existantes
    client_data = df_clients.loc[client_id].to_dict()

    # Compléter toutes les colonnes attendues par le pipeline
    full_input = {col: 0.0 for col in ALL_COLUMNS}
    for col in client_data:
        if col in ALL_COLUMNS:
            full_input[col] = client_data[col]

    df_input = pd.DataFrame([full_input])

    # Calcul prédiction
    proba = float(pipe.predict_proba(df_input)[0][1])
    decision = int(proba > THRESHOLD_METIER)

    return {
        "client_id": client_id,
        "score_probabilite": round(proba, 4),
        "decision": "Refusé" if decision == 1 else "Approuvé",
        "seuil": THRESHOLD_METIER
    }
