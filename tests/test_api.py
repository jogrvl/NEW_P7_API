# src/test_api.py

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

# 👉 Toujours utiliser le sample dans CI / Render
LOCAL_SAMPLE_PATH = os.path.join(BASE_DIR, "data", "train_df_sample.csv")

# 👉 Dataset complet uniquement en local
FULL_DATA_PATH = os.path.join(BASE_DIR, "..", "train_df_cleaned.csv")


# -----------------------------
# Chargement Dataset
# -----------------------------
def load_dataset():
    """
    Logique simple et fiable :
    1. Si train_df_sample.csv existe → on l’utilise (GitHub Actions / Render)
    2. Sinon si train_df_cleaned.csv existe → on l’utilise (local)
    3. Sinon → erreur claire (plus de Google Drive !)
    """

    # 1 → SAMPLE PRIORITAIRE (CI / Render)
    if os.path.exists(LOCAL_SAMPLE_PATH):
        print("➡ Chargement du SAMPLE local")
        return pd.read_csv(LOCAL_SAMPLE_PATH)

    # 2 → Full dataset pour travail local
    if os.path.exists(FULL_DATA_PATH):
        print("➡ Chargement du dataset complet local")
        return pd.read_csv(FULL_DATA_PATH)

    # 3 → Aucun fichier → erreur volontaire
    raise FileNotFoundError(
        "❌ Aucun dataset trouvé. Ajoutez train_df_sample.csv dans src/data/."
    )


# -----------------------------
# Chargement modèle + données
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ modele_pipeline.pkl est introuvable à la racine du projet.")

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

    # Remplit toutes les features (certaines peuvent manquer dans le sample)
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
        "seuil": THRESHOLD_METIER,
    }
