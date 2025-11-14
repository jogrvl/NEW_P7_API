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

# URL Google Drive (ID valide)
DRIVE_DOWNLOAD_URL = (
    "https://drive.google.com/uc?export=download&id=1LU8YL8FxHkYSCyG3cwQsguLcufz_Fm-J"
)


# -----------------------------
# Téléchargement gros fichier Google Drive
# -----------------------------
def download_big_file_from_google_drive(url, destination):
    """Télécharge un gros fichier Google Drive (avec token de confirmation)."""
    session = requests.Session()
    response = session.get(url, stream=True)

    # Chercher le token de confirmation (nécessaire pour les gros fichiers)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_url = url + "&confirm=" + value
            response = session.get(confirm_url, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_csv_if_needed():
    """Télécharge le CSV si non présent localement."""
    if os.path.exists(DATA_PATH):
        return

    print("➡ Téléchargement du dataset depuis Google Drive...")
    download_big_file_from_google_drive(DRIVE_DOWNLOAD_URL, DATA_PATH)
    print("✔ train_df_cleaned.csv téléchargé")


download_csv_if_needed()


# -----------------------------
# Chargement pipeline + données
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ modele_pipeline.pkl introuvable à la racine du projet.")

pipe = joblib.load(MODEL_PATH)

df_clients = pd.read_csv(DATA_PATH)
print("Colonnes trouvées dans le CSV :", df_clients.columns.tolist())

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

    # Vérifier si le client existe
    if client_id not in df_clients.index:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} non trouvé dans la base."
        )

    # Récupération des données du client
    client_data = df_clients.loc[client_id].to_dict()

    # Préparation des features attendues par le modèle
    full_input = {col: 0.0 for col in ALL_COLUMNS}
    for col in client_data:
        if col in ALL_COLUMNS:
            full_input[col] = client_data[col]

    df_input = pd.DataFrame([full_input])

    # Prédiction
    proba = float(pipe.predict_proba(df_input)[0][1])
    decision = int(proba > THRESHOLD_METIER)

    return {
        "client_id": client_id,
        "score_probabilite": round(proba, 4),
        "decision": "Refusé" if decision == 1 else "Approuvé",
        "seuil": THRESHOLD_METIER
    }
