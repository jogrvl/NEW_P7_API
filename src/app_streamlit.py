# src/app_streamlit.py

import streamlit as st
import requests

st.title("Test API - Scoring Crédit P7")

API_URL = st.text_input("URL de l'API (ex: http://127.0.0.1:8000/predict)", "")

# Champs principaux à remplir par l'utilisateur
DAYS_EMPLOYED = st.number_input("DAYS_EMPLOYED", -20000, 0, -3000)
AMT_INCOME_TOTAL = st.number_input("AMT_INCOME_TOTAL", 0, 1_000_000, 120000)
AMT_CREDIT = st.number_input("AMT_CREDIT", 0, 1_000_000, 200000)
APPROVED_DECISION_MAX = st.number_input("APPROVED_DECISION_MAX", 0.0, 1.0, 0.5)

if st.button("Obtenir la prédiction"):
    if not API_URL:
        st.error("Renseignez l'URL de l'API")
    else:
        # Créer le JSON à envoyer à l'API
        payload = {
            "DAYS_EMPLOYED": DAYS_EMPLOYED,
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
            "AMT_CREDIT": AMT_CREDIT,
            "APPROVED_DECISION_MAX": APPROVED_DECISION_MAX
        }
        try:
            r = requests.post(API_URL, json=payload)
            if r.ok:
                res = r.json()
                st.success(f"Probabilité : {res['score_probabilite']:.4f}")
                st.info(f"Décision : {res['prediction']}")
                st.text(f"Seuil utilisé : {res['seuil_utilise']}")
            else:
                st.error(f"Erreur API ({r.status_code})")
        except Exception as e:
            st.error(f"Erreur : {e}")
