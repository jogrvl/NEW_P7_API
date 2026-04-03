import streamlit as st
# import requests
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ============================================================
# CONFIGURATION GÉNÉRALE DE LA PAGE
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "modele_pipeline.pkl"
pipe = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="📊",
    layout="wide"
)

# URL de l'API locale
# API_URL = "http://127.0.0.1:8000/predict"

# Chemin vers le fichier de données clients
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train_df_sample.csv"


# ============================================================
# LABELS LISIBLES POUR L’INTERFACE
# ============================================================
HUMAN_LABELS = {
    "SK_ID_CURR": "ID Client",
    "TARGET": "Statut crédit",
    "CODE_GENDER": "Genre",
    "FLAG_OWN_CAR": "Voiture",
    "FLAG_OWN_REALTY": "Bien immobilier",
    "CNT_CHILDREN": "Enfants",
    "AMT_INCOME_TOTAL": "Revenu annuel",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Montant de l'annuité",
    "AMT_GOODS_PRICE": "Prix du bien",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_EDUCATION_TYPE": "Niveau d'éducation",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "NAME_HOUSING_TYPE": "Type de logement",
    "DAYS_BIRTH": "Âge (jours)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours)",
    "CNT_FAM_MEMBERS": "Membres du foyer",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
}

# Variables importantes à afficher dans le profil client
IMPORTANT_COLUMNS = [
    "SK_ID_CURR",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
]

# Variables proposées pour la comparaison avec la population
COMPARE_OPTIONS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

# Colonnes monétaires
MONEY_COLUMNS = {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
}

# Colonnes entières simples
INTEGER_COLUMNS = {
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "SK_ID_CURR",
}

# Colonnes en jours
DAY_COLUMNS = {
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
}


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def pretty_label(col: str) -> str:
    """Retourne un libellé lisible pour une colonne."""
    return HUMAN_LABELS.get(col, col)


def format_number_fr(value: float, decimals: int = 0) -> str:
    """
    Formate un nombre avec un style plus lisible :
    150000.0 -> 150 000
    1234.56 -> 1 234,56
    """
    formatted = f"{value:,.{decimals}f}"
    formatted = formatted.replace(",", " ").replace(".", ",")
    return formatted


def format_value(col: str, value):
    """
    Formate une valeur selon le type de variable :
    - euros
    - jours
    - entier
    - float arrondi
    """
    if pd.isna(value):
        return "Valeur manquante"

    if col in MONEY_COLUMNS:
        return f"{format_number_fr(float(value), 0)} €"

    if col in INTEGER_COLUMNS:
        return f"{int(round(float(value)))}"

    if col in DAY_COLUMNS:
        return f"{int(round(float(value)))} jours"

    if isinstance(value, float):
        return format_number_fr(value, 2)

    return str(value)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def load_data():
    """
    Charge le dataset client en mémoire.
    Le cache évite de relire le CSV à chaque interaction.
    """
    df = pd.read_csv(DATA_PATH)

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("La colonne SK_ID_CURR est absente du fichier.")

    return df


df_clients = load_data()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Paramètres")

client_id = st.sidebar.selectbox(
    "Sélectionnez un client",
    sorted(df_clients["SK_ID_CURR"].unique())
)

compare_var = st.sidebar.selectbox(
    "Variable de comparaison",
    COMPARE_OPTIONS,
    format_func=pretty_label
)

launch_prediction = st.sidebar.button("Obtenir la prédiction")


# ============================================================
# DONNÉES DU CLIENT SÉLECTIONNÉ
# ============================================================
client_data = df_clients[df_clients["SK_ID_CURR"] == client_id].copy()


# ============================================================
# EN-TÊTE PRINCIPAL
# ============================================================
st.title("📊 Dashboard Scoring Crédit")
st.caption("Application d’aide à la décision pour l’évaluation du risque client")

st.markdown("---")


# ============================================================
# RÉSULTAT DU SCORING
# ============================================================
if launch_prediction:
    try:
        # On récupère la ligne du client sans l'identifiant
        client_row = client_data.drop(columns=["SK_ID_CURR"])

        # Colonnes attendues par le pipeline
        model_columns = pipe.feature_names_in_

        # Construire un input complet avec toutes les colonnes attendues
        full_input = {col: 0.0 for col in model_columns}
        for col in client_row.columns:
            if col in model_columns:
                full_input[col] = client_row.iloc[0][col]

        # Transformer en DataFrame pour le modèle
        df_input = pd.DataFrame([full_input])

        # Prédiction
        proba = float(pipe.predict_proba(df_input)[0][1])
        prediction = "Refusé" if proba > 0.54 else "Approuvé"

        result = {
            "client_id": int(client_id),
            "score_probabilite": proba,
            "prediction": prediction,
            "seuil_utilise": 0.54
        }

        st.subheader("🎯 Résultat du scoring")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ID Client", f"{result['client_id']}")

        with col2:
            st.metric(
                "Probabilité de défaut",
                f"{result['score_probabilite']:.2%}"
            )

        with col3:
            st.metric(
                "Seuil utilisé",
                f"{result['seuil_utilise']:.2%}"
            )

        if result["prediction"] == "Approuvé":
            st.success("✅ Décision estimée : Approuvé")
        else:
            st.error("❌ Décision estimée : Refusé")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction locale : {e}")

else:
    st.info("Sélectionnez un client dans la barre latérale puis lancez la prédiction.")


st.markdown("---")


# ============================================================
# AFFICHAGE PRINCIPAL : PROFIL CLIENT + COMPARAISON
# ============================================================
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("📄 Profil du client")

    # On ne garde que les colonnes utiles disponibles
    available_columns = [col for col in IMPORTANT_COLUMNS if col in client_data.columns]

    # Transformation pour affichage vertical
    client_display = client_data[available_columns].T.reset_index()
    client_display.columns = ["Variable", "Valeur"]

    # Conserver le nom technique temporairement pour le formatage
    client_display["Variable brute"] = client_display["Variable"]

    # Appliquer libellés lisibles
    client_display["Variable"] = client_display["Variable"].map(pretty_label)

    # Appliquer formatage des valeurs
    client_display["Valeur"] = client_display.apply(
        lambda row: format_value(row["Variable brute"], row["Valeur"]),
        axis=1
    )

    # Garder uniquement les colonnes utiles à l’affichage
    client_display = client_display[["Variable", "Valeur"]]

    st.dataframe(client_display, use_container_width=True, hide_index=True)

with right_col:
    st.subheader("📈 Comparaison avec la population")

    fig = px.histogram(
        df_clients,
        x=compare_var,
        nbins=40,
        title=f"Distribution - {pretty_label(compare_var)}"
    )

    client_value = client_data.iloc[0][compare_var]

    fig.add_vline(
        x=client_value,
        line_dash="dash",
        annotation_text="Client",
        annotation_position="top right"
    )

    fig.update_layout(
        xaxis_title=pretty_label(compare_var),
        yaxis_title="Nombre de clients"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Valeur du client sélectionné**")
    st.info(f"{pretty_label(compare_var)} : {format_value(compare_var, client_value)}")


st.markdown("---")


# ============================================================
# DONNÉES BRUTES (OPTIONNEL)
# ============================================================
with st.expander("Afficher toutes les données brutes du client"):
    st.dataframe(client_data, use_container_width=True)