# Projet 7 - Scoring Crédit

## Contexte

Projet réalisé dans le cadre de la formation Data Science OpenClassrooms.
Développement d'un modèle de scoring crédit pour la société « Prêt à dépenser »,
avec une démarche MLOps complète (de la modélisation au déploiement continu et
au suivi du data drift).

## Objectif

Prédire la probabilité de défaut de paiement d'un client et classifier les
demandes de crédit (accordé / refusé) via un seuil optimisé sur un coût métier.
Le modèle est exposé sous forme d'API déployée sur le cloud.

## API en ligne

Déployée sur Hugging Face Spaces : **https://rmercierwork-scoring-credit-p7.hf.space**

- `GET /health` : vérifie que l'API est en ligne
- `POST /predict` : renvoie la probabilité de défaut, le seuil et la décision

## Structure du projet

```
Projet_7/
├── api/                # API de prédiction
│   ├── app.py                # application Flask (/predict, /health)
│   ├── requirements.txt      # packages de l'API
│   └── test_app.py           # tests unitaires (pytest)
├── notebooks/
│   ├── 01_preparation_donnees.ipynb   # EDA, feature engineering, préprocessing
│   ├── 02_modelisation.ipynb          # modèles, score métier, MLFlow, SHAP
│   └── 03_data_drift.ipynb            # analyse de data drift (Evidently)
├── models/             # modèles sauvegardés (lgbm_final.pkl, seuil_optimal.pkl)
├── mlruns/             # tracking MLFlow (non versionné)
├── .github/workflows/  # pipeline CI/CD (tests + déploiement)
├── Dockerfile          # conteneur de déploiement de l'API
├── requirements.txt    # packages du projet
└── README.md
```

## Démarche MLOps

- **Tracking des expérimentations** via MLFlow (un run par modèle : params, métriques, modèle)
- **Stockage centralisé** du modèle final dans le Model Registry MLFlow (`scoring_credit_lgbm`)
- **Déploiement continu** via GitHub Actions : tests unitaires (pytest) puis déploiement
  automatique de l'API sur Hugging Face Spaces
- **Suivi en production** : analyse de data drift avec Evidently (train vs test)

## Modélisation

- Feature engineering : agrégation des tables annexes, sélection des 25 variables
  les plus prédictives
- Score métier : coût asymétrique (faux négatif ×10, faux positif ×1) + optimisation
  du seuil de décision (0.52)
- Gestion du déséquilibre des classes (`class_weight = balanced`)
- Interprétabilité globale et locale via SHAP

| Modèle | AUC | Coût métier | Seuil |
|--------|-----|-------------|-------|
| DummyClassifier (baseline) | 0.500 | 1.000 | — |
| Régression Logistique | 0.735 | 0.692 | 0.52 |
| LightGBM baseline | 0.764 | 0.644 | 0.50 |
| LightGBM optimisé | 0.764 | 0.643 | 0.52 |

## Stack technique

- Python 3.10
- pandas, numpy, scikit-learn, LightGBM, SHAP
- MLFlow (tracking + model registry)
- Flask + Docker pour l'API
- Evidently pour le data drift
- GitHub Actions pour le CI/CD
- Git / GitHub pour le versioning

## Installation

```bash
python -m venv P7env
.\P7env\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'API en local

```bash
cd api
pip install -r requirements.txt
python app.py        # http://localhost:7860
```

## Lancer les tests

```bash
pytest api/test_app.py -v
```
