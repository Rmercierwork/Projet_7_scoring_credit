# ============================================================
# API Flask - Scoring Crédit
# ============================================================
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialisation de l'application Flask
app = Flask(__name__)

# Chemin absolu vers les modèles (indépendant du répertoire de lancement)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_final.pkl')
SEUIL_PATH = os.path.join(BASE_DIR, 'models', 'seuil_optimal.pkl')

# Chargement du modèle et du seuil au démarrage
model = joblib.load(MODEL_PATH)
seuil = joblib.load(SEUIL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction.
    Reçoit un JSON avec les features du client,
    retourne la probabilité de défaut et la décision.
    """
    try:
        # Récupération des données du client
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({'error': 'Aucune donnée reçue'}), 400

        # Conversion en DataFrame
        client_df = pd.DataFrame([data])

        # Prédiction
        proba = model.predict_proba(client_df)[0][1]
        decision = 'REFUSÉ' if proba >= seuil else 'ACCORDÉ'

        return jsonify({
            'probabilite_defaut': round(float(proba), 4),
            'seuil': round(float(seuil), 4),
            'decision': decision
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint de vérification que l'API est bien en ligne.
    """
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5001)