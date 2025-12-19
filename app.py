# ---------------------------
# app.py
# ---------------------------

import streamlit as st
st.set_page_config(page_title="Expresso Churn Predictor", page_icon="📞")  # DOIT ÊTRE EN PREMIER
import pandas as pd
import joblib
import numpy as np

# ---------------------------
# CHARGEMENT DU MODÈLE ET DES LABEL ENCODERS
# ---------------------------
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('expresso_churn_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        # Les colonnes utilisées par le modèle
        feature_names = [
            'REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
            'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE',
            'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK'
        ]
        return model, label_encoders, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant : {e}")
        st.stop()

model, label_encoders, FEATURE_NAMES = load_model_and_encoders()

# Colonnes catégorielles à encoder
CAT_COLS = ['REGION', 'TENURE', 'MRG', 'TOP_PACK']

# ---------------------------
# INTERFACE STREAMLIT
# ---------------------------
st.title("📞 Prédiction de désabonnement - Expresso")
st.markdown("Remplissez les caractéristiques du client pour prédire son risque de désabonnement.")

# ---------------------------
# SAISIE DES DONNÉES
# ---------------------------
input_data = {}

# Options pour certaines variables
TENURE_OPTIONS = ['3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24', '>24']
MRG_OPTIONS = ['NO', 'YES']

for col in FEATURE_NAMES:
    if col == 'TENURE':
        val = st.selectbox("Durée d'abonnement (TENURE)", options=TENURE_OPTIONS)
        input_data[col] = val
    elif col == 'MRG':
        val = st.selectbox("Fusion avec un autre opérateur ? (MRG)", options=MRG_OPTIONS)
        input_data[col] = val
    elif col in ['REGION', 'TOP_PACK']:
        classes = sorted(label_encoders[col].classes_.tolist())
        val = st.selectbox(f"{col}", options=classes)
        input_data[col] = val
    else:
        input_data[col] = st.number_input(
            f"{col}",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f"
        )

# ---------------------------
# PRÉDICTION
# ---------------------------
if st.button("🔍 Prédire le risque de churn"):
    try:
        # Créer le DataFrame
        df_input = pd.DataFrame([input_data])

        # Encoder les variables catégorielles
        for col in CAT_COLS:
            original_val = str(df_input[col].iloc[0])
            le = label_encoders[col]

            if original_val in le.classes_:
                encoded_val = le.transform([original_val])[0]
            else:
                st.warning(
                    f"⚠️ Valeur '{original_val}' pour '{col}' non vue pendant l'entraînement. "
                    "Utilisation de la catégorie la plus courante."
                )
                encoded_val = 0
            df_input[col] = encoded_val

        # Ordre des colonnes et conversion en float
        df_input = df_input[FEATURE_NAMES].astype(float)

        # Prédiction
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]

        # Affichage du résultat
        st.subheader("Résultat de la prédiction")
        if prediction == 1:
            st.error("🔴 Risque élevé de désabonnement !")
        else:
            st.success("🟢 Client fidèle (faible risque).")

        # Affichage des probabilités
        if len(proba) == 2:
            st.metric("Probabilité de désabonnement", f"{proba[1]:.2%}")
            st.metric("Probabilité de fidélité", f"{proba[0]:.2%}")
        else:
            st.warning("Le modèle ne renvoie pas deux classes, impossible d’afficher les probabilités.")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        st.write("Vérifiez que tous les fichiers sont présents et que le modèle a été entraîné correctement.")
