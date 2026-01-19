import pytest

from app import predict

def test_load_latest_model_missing_dir(monkeypatch, tmp_path):
    missing_dir = tmp_path / "missing_model_dir"
    monkeypatch.setattr(predict, "MODEL_DIR", missing_dir)
    with pytest.raises(FileNotFoundError):
        predict.load_latest_model()

def test_load_latest_model_empty_dir(monkeypatch, tmp_path):
    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(predict, "MODEL_DIR", empty_dir)
    with pytest.raises(FileNotFoundError):
        predict.load_latest_model()