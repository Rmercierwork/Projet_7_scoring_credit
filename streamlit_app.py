# ============================================================
# Interface Streamlit - Scoring Crédit
# ============================================================
import streamlit as st
import requests
import pandas as pd
import json

# Configuration de la page
st.set_page_config(
    page_title="Scoring Crédit - Prêt à dépenser",
    page_icon="💳",
    layout="centered"
)

# URL de l'API
API_URL = "https://projet-7-scoring-credit-apst.onrender.com"

# Titre
st.title("💳 Scoring Crédit")
st.subheader("Prêt à dépenser — Outil de décision crédit")
st.markdown("---")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("data/application_train_prepared.csv")
    return df

df = load_data()

# Sélection du client
st.header("Sélection du client")
client_ids = df['SK_ID_CURR'].tolist()
selected_id = st.selectbox("Choisissez un ID client :", client_ids)

# Récupération des features du client sélectionné
client_data = df[df['SK_ID_CURR'] == selected_id].drop(
    columns=['TARGET', 'SK_ID_CURR']
).iloc[0].to_dict()

if st.button("Calculer le score", type="primary"):
    with st.spinner("Calcul en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=client_data,
                timeout=60
            )
            result = response.json()

            st.markdown("---")
            st.header("Résultat")

            # Affichage de la décision
            decision = result['decision']
            proba = result['probabilite_defaut']
            seuil = result['seuil']

            if decision == 'ACCORDÉ':
                st.success(f"✅ Crédit **{decision}**")
            else:
                st.error(f"❌ Crédit **{decision}**")

            # Métriques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probabilité de défaut", f"{proba:.1%}")
            with col2:
                st.metric("Seuil de décision", f"{seuil:.1%}")
            with col3:
                st.metric("ID Client", selected_id)

            # Jauge visuelle
            st.markdown("### Niveau de risque")
            st.progress(proba)

        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")