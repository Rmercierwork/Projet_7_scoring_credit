# Projet 7 - Scoring Crédit

## Contexte
Projet réalisé dans le cadre de la formation Data Science OpenClassrooms.
Développement d'un modèle de scoring crédit pour la société "Prêt à dépenser".

## Objectif
Prédire la probabilité de défaut de paiement d'un client et classifier
les demandes de crédit (accordé/refusé) via un seuil optimisé métier.

## Structure du projet

```
Projet_7/
├── data/               # Données brutes et préparées (non versionnées)
├── notebooks/
│   ├── 01_preparation_donnees.ipynb   # EDA, feature engineering, préprocessing
│   └── 02_modelisation.ipynb          # Modèles, score métier, SHAP
├── models/             # Modèles sauvegardés (non versionnés)
├── mlruns/             # Tracking MLFlow (non versionné)
└── README.md
```

## Modèles entraînés

| Modèle | AUC | Coût métier |
|--------|-----|-------------|
| DummyClassifier (baseline) | 0.50 | 1.00 |
| Régression Logistique | 0.76 | 0.65 |
| LightGBM optimisé | 0.78 | 0.62 |

## Stack technique
- Python 3.10
- pandas, numpy, scikit-learn, LightGBM, SHAP
- MLFlow pour le tracking des expérimentations
- Git/GitHub pour le versioning

## Installation

```bash
python -m venv P7env
.\P7env\Scripts\activate
pip install -r requirements.txt
```
