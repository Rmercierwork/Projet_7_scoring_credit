# ============================================================
# Tests unitaires - API Flask Scoring Crédit
# ============================================================
import pytest
import json
import sys
import os

# On ajoute le dossier api au path pour pouvoir importer app
sys.path.insert(0, os.path.dirname(__file__))
from app import app

@pytest.fixture
def client():
    """Crée un client de test Flask."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    """Test que l'endpoint /health répond correctement."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'


def test_predict_valid(client):
    """Test que l'endpoint /predict répond avec les bonnes clés."""
    # Données minimales pour un test (valeurs neutres)
    fake_client = {f"feature_{i}": 0.0 for i in range(254)}

    response = client.post(
        '/predict',
        json=fake_client,
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'probabilite_defaut' in data
    assert 'seuil' in data
    assert 'decision' in data


def test_predict_no_data(client):
    """Test que l'API gère correctement une requête sans données."""
    response = client.post('/predict', content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_decision_type(client):
    """Test que la décision est bien ACCORDÉ ou REFUSÉ."""
    fake_client = {f"feature_{i}": 0.0 for i in range(254)}
    response = client.post('/predict', json=fake_client)
    data = json.loads(response.data)
    assert data['decision'] in ['ACCORDÉ', 'REFUSÉ']


def test_predict_probabilite_range(client):
    """Test que la probabilité est bien entre 0 et 1."""
    fake_client = {f"feature_{i}": 0.0 for i in range(254)}
    response = client.post('/predict', json=fake_client)
    data = json.loads(response.data)
    assert 0.0 <= data['probabilite_defaut'] <= 1.0