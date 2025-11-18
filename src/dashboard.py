# dashboard.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://new-p7-api.onrender.com/predict"

CSV_PATH = r"C:\Users\jogrv\NEW_P7_API\src\data\train_df_sample.csv"

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    layout="wide",
)

# ============================================================
# LABELS LISIBLES POUR LES VARIABLES
# ============================================================

HUMAN_LABELS = {
    "SK_ID_CURR": "ID Client",
    "TARGET": "Statut du crédit",
    "CODE_GENDER": "Genre",
    "FLAG_OWN_CAR": "Possession d'une voiture",
    "FLAG_OWN_REALTY": "Possession d'un logement",
    "CNT_CHILDREN": "Nombre d'enfants",
    "AMT_INCOME_TOTAL": "Revenu total annuel (€)",
    "AMT_CREDIT": "Montant du crédit demandé (€)",
    "AMT_ANNUITY": "Montant de l'annuité (€)",
    "AMT_GOODS_PRICE": "Prix du bien (€)",
    "NAME_TYPE_SUITE": "Type de famille",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_EDUCATION_TYPE": "Niveau d'éducation",
    "NAME_FAMILY_STATUS": "Statut familial",
    "NAME_HOUSING_TYPE": "Type de logement",
    "DAYS_BIRTH": "Âge en jours (négatif)",
    "DAYS_EMPLOYED": "Ancienneté professionnelle (jours, négatif)",
    "DAYS_REGISTRATION": "Ancienneté de l'enregistrement (jours, négatif)",
    "DAYS_ID_PUBLISH": "Ancienneté du document d'identité (jours, négatif)",
    "OWN_CAR_AGE": "Âge de la voiture",
    "FLAG_MOBIL": "Possède un téléphone mobile",
    "FLAG_EMP_PHONE": "Possède un téléphone pro",
    "FLAG_WORK_PHONE": "Possède un téléphone travail",
    "FLAG_CONT_MOBILE": "Contrat mobile actif",
    "FLAG_PHONE": "Possède un téléphone",
    "FLAG_EMAIL": "Possède un email",
    "OCCUPATION_TYPE": "Type d'emploi",
    "CNT_FAM_MEMBERS": "Nombre de membres dans la famille",
    "REGION_POPULATION_RELATIVE": "Proportion de population dans la région",
    "DAYS_LAST_PHONE_CHANGE": "Dernier changement de téléphone (jours)",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Demandes de crédit dernière heure",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Demandes de crédit dernier jour",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Demandes de crédit dernière semaine",
    "AMT_REQ_CREDIT_BUREAU_MON": "Demandes de crédit dernier mois",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Demandes de crédit dernier trimestre",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Demandes de crédit dernière année",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "POS_NAME_CONTRACT_STATUS_XNA_MEAN": "Statut contrat XNA (moyenne)",
    "POS_NAME_CONTRACT_STATUS_nan_MEAN": "Statut contrat NaN (moyenne)",
    "INS_PAYMENT_PERC_MEAN": "Paiement % moyen",
    "INS_PAYMENT_PERC_VAR": "Variance du paiement %",
    "INS_PAYMENT_DIFF_MEAN": "Différence moyenne paiement",
    "INS_PAYMENT_DIFF_VAR": "Variance différence paiement",
    "INS_DPD_MAX": "Nombre de jours de retard max",
    "INS_DPD_MEAN": "Nombre de jours de retard moyen",
    "INS_DBD_MAX": "Nombre de jours de défaut max",
    "INS_DBD_MEAN": "Nombre de jours de défaut moyen",
}

def pretty(col):
    return HUMAN_LABELS.get(col, col.replace("_", " ").title())

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.set_index("SK_ID_CURR", inplace=True)
    return df

df_clients = load_data()

# ============================================================
# TITRE
# ============================================================

st.title("📊 Dashboard Scoring Crédit")
st.markdown("Outil interactif pour les chargés de relation client — Version 1")

# ============================================================
# SÉLECTION DU CLIENT
# ============================================================

client_id = st.selectbox(
    "Sélectionnez un client :", 
    df_clients.index.sort_values()
)

client_data = df_clients.loc[client_id]

st.markdown("---")

# ============================================================
# FONCTION DE REPLI LOCAL
# ============================================================

def fallback_prediction(df, client_id):
    row = df.loc[client_id]
    score = 0.15  # valeur par défaut
    decision = "Refusé" if score > 0.54 else "Approuvé"
    return {
        "client_id": client_id,
        "score_probabilite": score,
        "decision": decision,
        "seuil": 0.54
    }

# ============================================================
# APPEL À L'API POUR LE SCORE
# ============================================================

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("📝 Obtenir la prédiction du modèle"):

        payload = {"SK_ID_CURR": int(client_id)}

        try:
            response = requests.post(API_URL, json=payload, timeout=20)
            response.raise_for_status()
            st.session_state["prediction"] = response.json()

        except Exception as e:
            st.error(f"❌ Erreur API : {e} — fallback local activé.")
            st.session_state["prediction"] = fallback_prediction(df_clients, client_id)

# ============================================================
# AFFICHAGE DU SCORE
# ============================================================

if "prediction" in st.session_state:
    pred = st.session_state["prediction"]

    with col2:
        st.subheader("🎯 Résultat du modèle")

        score = pred["score_probabilite"]
        seuil = pred["seuil"]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Probabilité de défaut"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': 'darkred' if score > seuil else 'green'},
                'steps': [
                    {'range': [0, seuil], 'color': '#b6e3b6'},
                    {'range': [seuil, 1], 'color': '#f5b5b5'},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'value': seuil,
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if score < seuil:
            st.success("Client Approuvé – faible risque estimé.")
        else:
            st.error("Client Refusé – risque estimé trop élevé.")

st.markdown("---")

# ============================================================
# INFORMATIONS DU CLIENT
# ============================================================

st.subheader("📄 Informations essentielles du client")

important_vars = [
    "AMT_INCOME_TOTAL",
    "CNT_CHILDREN",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

df_display = client_data[important_vars].rename(pretty)
st.dataframe(df_display)

st.markdown("---")

# ============================================================
# COMPARAISON AVEC LES AUTRES CLIENTS
# ============================================================

st.subheader("📈 Comparaison avec l'ensemble des clients")

column_to_compare = st.selectbox(
    "Variable à comparer :",
    df_clients.columns,
    format_func=pretty,
)

fig2 = px.histogram(
    df_clients,
    x=column_to_compare,
    nbins=40,
    opacity=0.7,
    labels={column_to_compare: pretty(column_to_compare)}
)

fig2.add_vline(
    x=client_data[column_to_compare],
    line_dash="dash",
    line_color="red",
    annotation_text="Client",
)

st.plotly_chart(fig2, use_container_width=True)
