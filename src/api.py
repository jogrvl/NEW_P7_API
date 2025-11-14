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

# Pipeline
MODEL_PATH = os.path.join(BASE_DIR, "..", "modele_pipeline.pkl")

# 👉 Nouveau : on utilise un sample local pour GitHub
LOCAL_SAMPLE_PATH = os.path.join(BASE_DIR, "data", "train_df_sample.csv")

# 👉 Fichier complet (Render ou local hors GitHub)
FULL_DATA_PATH = os.path.join(BASE_DIR, "..", "train_df_cleaned.csv")

# URL Google Drive
DRIVE_DOWNLOAD_URL = (
    "https://drive.google.com/uc?export=download&id=1LU8YL8FxHkYSCyG3cwQsguLcufz_Fm-J"
)


# -----------------------------
# Téléchargement Google Drive
# -----------------------------
def download_big_file_from_google_drive(url, destination):
    session = requests.Session()
    response = session.get(url, stream=True)

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_url = url + "&confirm=" + value
            response = session.get(confirm_url, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def load_dataset():
    """
    Charge le dataset selon le contexte :
      1. Si le sample existe → on l'utilise (GitHub Actions)
      2. Sinon si le full existe → on le charge (local)
      3. Sinon → télécharger depuis Google Drive (Render)
    """

    # 1. Sample présent → PRIORITAIRE pour GitHub Actions
    if os.path.exists(LOCAL_SAMPLE_PATH):
        print("➡ Chargement du SAMPLE local (GitHub Actions)")
        return pd.read_csv(LOCAL_SAMPLE_PATH)

    # 2. Full dataset local
    if os.path.exists(FULL_DATA_PATH):
        print("➡ Chargement du dataset complet local")
        return pd.read_csv(FULL_DATA_PATH)

    # 3. Render → téléchargement automatique
    print("➡ Téléchargement du dataset complet depuis Google Drive...")
    download_big_file_from_google_drive(DRIVE_DOWNLOAD_URL, FULL_DATA_PATH)
    print("✔ Dataset complet téléchargé")
    return pd.read_csv(FULL_DATA_PATH)


# -----------------------------
# Chargement modèle + données
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ modele_pipeline.pkl introuvable à la racine du projet.")

pipe = joblib.load(MODEL_PATH)

df_clients = load_dataset()
print("Colonnes trouvées :", df_clients.columns.tolist())

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


@app.post("/predict")
def predict(request: ClientRequest):

    client_id = request.SK_ID_CURR

    if client_id not in df_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")

    client_data = df_clients.loc[client_id].to_dict()

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
