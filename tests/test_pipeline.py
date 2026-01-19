# tests/test_pipeline.py
import pathlib
import json

import pandas as pd
import pytest

from app.pipeline import load_json_dataset, feature_engineering, prepare_training_data, build_training_pipeline, train_and_save

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"

@pytest.fixture
def sample_json(tmp_path):
    dates = [
        "01/01/2020",
        "08/01/2020",
        "15/01/2020",
        "22/01/2020",
        "29/01/2020",
        "05/02/2020",
        "12/02/2020",
        "19/02/2020",
    ]
    data = []
    for i, date_str in enumerate(dates):
        offset = i * 6
        dezenas = [f"{n:02d}" for n in range(1 + offset, 7 + offset)]
        data.append(
            {
                "loteria": "megasena",
                "concurso": 9999 + i,
                "data": date_str,
                "local": "Auditório",
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
    fp = tmp_path / "sample.json"
    fp.write_text(json.dumps(data))
    return fp

def test_load_json_dataset(tmp_path):
    data = [
        None,
        {
            "loteria": "megasena",
            "concurso": 1,
            "data": "01/01/2020",
            "dezenas": ["01","02","03","04","05","06"],
            "acumulou": False,
            "valorArrecadado": 1000,
            "valorAcumuladoConcurso_0_5": 0,
            "valorAcumuladoConcursoEspecial": 0,
            "valorAcumuladoProximoConcurso": 0,
            "valorEstimadoProximoConcurso": 3000000,
        },
        None,
        {
            "loteria": "megasena",
            "concurso": 2,
            "data": "08/01/2020",
            "dezenas": ["10","20","30","40","50","60"],
            "acumulou": True,
            "valorArrecadado": 2000,
            "valorAcumuladoConcurso_0_5": 0,
            "valorAcumuladoConcursoEspecial": 0,
            "valorAcumuladoProximoConcurso": 0,
            "valorEstimadoProximoConcurso": 3000000,
        },
    ]
    fp = tmp_path / "dataset.json"
    fp.write_text(json.dumps(data))

    df = load_json_dataset(fp)
    assert df.shape[0] == 2
    assert "concurso" in df.columns

def test_feature_engineering(sample_json):
    raw = load_json_dataset(sample_json)
    df = feature_engineering(raw)
    
    assert "year" in df.columns and df["year"].iloc[0] == 2020
    assert "sum_dezenas" in df.columns and df["sum_dezenas"].iloc[0] == 21
    assert "target" not in df.columns


def test_build_training_pipeline():
    pipe = build_training_pipeline()
    assert list(pipe.named_steps) == ["preprocess", "clf"]


def test_prepare_training_data_shift_and_normalize():
    df = pd.DataFrame(
        {
            "concurso": [3, 1, 4, 2],
            "acumulou": ["sim", True, "nao", "false"],
            "valorArrecadado": [300.0, 100.0, 400.0, 200.0],
        }
    )

    X, y = prepare_training_data(df)

    assert list(X["concurso"]) == [1, 2, 3]
    assert list(y) == [0, 1, 0]
    assert "target" not in X.columns
    assert "acumulou_int" not in X.columns


def test_train_and_save(tmp_path, sample_json):
    model_dir = tmp_path / "model"
    pipeline, version = train_and_save(sample_json, model_dir, version="testv1")
    
    assert (model_dir / f"model_{version}.pkl").exists()
    
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["version"] == version
    
    assert hasattr(pipeline.named_steps["clf"], "classes_")
