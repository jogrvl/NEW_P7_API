from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Paramètres
# -----------------------------
THRESHOLD_METIER = 0.54

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "modele_pipeline.pkl"
DATA_PATH = PROJECT_DIR / "data" / "train_df_sample.csv"

# -----------------------------
# Vérifications fichiers
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Modèle introuvable : {MODEL_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset introuvable : {DATA_PATH}")

# -----------------------------
# Chargement modèle + données
# -----------------------------
pipe = joblib.load(MODEL_PATH)

df_clients = pd.read_csv(DATA_PATH)

if "SK_ID_CURR" not in df_clients.columns:
    raise KeyError("❌ La colonne 'SK_ID_CURR' est absente du dataset.")

df_clients.set_index("SK_ID_CURR", inplace=True)

ALL_COLUMNS = pipe.feature_names_in_

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="API Scoring Crédit P7",
    version="1.1",
    description="API de prédiction de risque crédit à partir de SK_ID_CURR"
)

class ClientRequest(BaseModel):
    SK_ID_CURR: int

@app.get("/")
def root():
    return {"message": "API Scoring Crédit - OK"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/clients")
def get_clients():
    return {"clients": df_clients.index.tolist()}

@app.post("/predict")
def predict(request: ClientRequest):
    client_id = request.SK_ID_CURR

    if client_id not in df_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")

    client_data = df_clients.loc[client_id].to_dict()

    # Préparer toutes les colonnes attendues par le pipeline
    full_input = {col: 0.0 for col in ALL_COLUMNS}
    for col, value in client_data.items():
        if col in ALL_COLUMNS:
            full_input[col] = value

    df_input = pd.DataFrame([full_input])

    proba = float(pipe.predict_proba(df_input)[0][1])
    prediction = "Refusé" if proba > THRESHOLD_METIER else "Approuvé"

    return {
        "client_id": int(client_id),
        "score_probabilite": round(proba, 4),
        "prediction": prediction,
        "seuil_utilise": THRESHOLD_METIER
    }