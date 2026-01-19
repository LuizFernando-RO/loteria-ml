# tests/test_api.py
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import predict
from app.pipeline import train_and_save

client = TestClient(app)

@pytest.fixture
def trained_model(tmp_path, monkeypatch):
    """Train a tiny model once for the whole test session."""
    data_path = tmp_path / "megasae_sample.json"
    dates = [
        "01/01/2021",
        "08/01/2021",
        "15/01/2021",
        "22/01/2021",
        "29/01/2021",
        "05/02/2021",
        "12/02/2021",
        "19/02/2021",
    ]
    sample = []
    for i, date_str in enumerate(dates):
        offset = i * 6
        dezenas = [f"{n:02d}" for n in range(1 + offset, 7 + offset)]
        sample.append(
            {
                "loteria": "megasena",
                "concurso": 1000 + i,
                "data": date_str,
                "local": "Auditorio",
                "dezenasOrdemSorteio": dezenas,
                "dezenas": dezenas,
                "acumulou": i % 2 == 0,
                "valorArrecadado": 100000 + (i * 10000),
                "valorAcumuladoConcurso_0_5": 0,
                "valorAcumuladoConcursoEspecial": 0,
                "valorAcumuladoProximoConcurso": 0,
                "valorEstimadoProximoConcurso": 3000000,
            }
        )
    data_path.write_text(json.dumps(sample))

    model_dir = tmp_path / "model"
    train_and_save(data_path, model_dir, version="unittest")
    monkeypatch.setattr(predict, "MODEL_DIR", model_dir)
    monkeypatch.setattr(predict, "METADATA_PATH", model_dir / "metadata.json")
    return model_dir

def test_predict_success(trained_model):
    payload = {
        "loteria": "megasena",
        "concurso": 2000,
        "data": "15/08/2022",
        "local": "Auditório",
        "dezenasOrdemSorteio": ["10", "20", "30", "40", "50", "60"],
        "dezenas": ["10", "20", "30", "40", "50", "60"],
        "acumulou": None,
        "valorArrecadado": 200000,
        "valorAcumuladoConcurso_0_5": 0,
        "valorAcumuladoConcursoEspecial": 0,
        "valorAcumuladoProximoConcurso": 0,
        "valorEstimadoProximoConcurso": 3000000,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data
    assert "timestamp" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["prediction"], bool)

def test_predict_success_list(trained_model):
    payload = [
        {
            "loteria": "megasena",
            "concurso": 2000,
            "data": "15/08/2022",
            "local": "Auditório",
            "dezenasOrdemSorteio": ["10", "20", "30", "40", "50", "60"],
            "dezenas": ["10", "20", "30", "40", "50", "60"],
            "acumulou": None,
            "valorArrecadado": 200000,
            "valorAcumuladoConcurso_0_5": 0,
            "valorAcumuladoConcursoEspecial": 0,
            "valorAcumuladoProximoConcurso": 0,
            "valorEstimadoProximoConcurso": 3000000,
        },
        {
            "loteria": "megasena",
            "concurso": 2001,
            "data": "22/08/2022",
            "local": "Auditório",
            "dezenasOrdemSorteio": ["11", "21", "31", "41", "51", "61"],
            "dezenas": ["11", "21", "31", "41", "51", "61"],
            "acumulou": None,
            "valorArrecadado": 210000,
            "valorAcumuladoConcurso_0_5": 0,
            "valorAcumuladoConcursoEspecial": 0,
            "valorAcumuladoProximoConcurso": 0,
            "valorEstimadoProximoConcurso": 3000000,
        },
    ]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert 0.0 <= data[0]["probability"] <= 1.0
    assert isinstance(data[0]["prediction"], bool)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_invalid_payload():
    payload = {
        "loteria": "megasena",
        "data": "15/08/2022",
        "local": "Auditório",
        "dezenas": ["10","20","30","40","50","60"],
        "acumulou": None,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
