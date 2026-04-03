# Credit Scoring Dashboard

Application data de scoring crédit construite autour d’un pipeline de prédiction, d’une API FastAPI et d’un dashboard Streamlit.

## Objectif du projet

Ce projet a pour but de proposer un outil d’aide à la décision pour évaluer le risque de défaut d’un client à partir d’un identifiant client (`SK_ID_CURR`).

L’application permet :
- de charger un client depuis un échantillon de données,
- d’interroger un modèle de scoring via une API,
- d’afficher la probabilité de défaut,
- de visualiser certaines informations client et de les comparer à la population.

## Fonctionnalités

- API FastAPI pour exposer la prédiction
- Dashboard Streamlit interactif
- Modèle de scoring sauvegardé avec `joblib`
- Sélection d’un client à partir de son identifiant
- Affichage du score de probabilité de défaut
- Comparaison visuelle avec la population
- Tests API

## Stack technique

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- Plotly
- Pytest
- Joblib

## Structure du projet

```text
NEW_P7_API/
│
├── app/
│   └── streamlit_app.py
├── data/
│   └── train_df_sample.csv
├── models/
│   └── modele_pipeline.pkl
├── notebook/
│   └── ...
├── src/
│   ├── __init__.py
│   └── api.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── requirements.txt
└── README.md