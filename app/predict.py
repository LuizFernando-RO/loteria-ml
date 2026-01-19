import pathlib
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from .pipeline import feature_engineering

MODEL_DIR = pathlib.Path(__file__).parent / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"


class PredictionResult(BaseModel):
    prediction: bool = Field(..., description="True = acumulou")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidade de acumular")
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


def load_latest_model() -> tuple:
    """Carrega o *.pkl mais recente do modelo em MODEL_DIR."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Diretório de modelo não existe")

    pkl_files = sorted(MODEL_DIR.glob("model_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError("Nenhum modelo encontrado")

    latest = pkl_files[-1]
    logger.info(f"Carregando o modelo de: {latest}")
    model = joblib.load(latest)

    version = latest.stem.split("_", 1)[1]
    return model, version

def predict_from_raw(payload: Dict[str, Any]) -> PredictionResult:
    """
    Obtém uma entrada em formato JSON, aplica o pipeline de feature engineering
    utilizado para treinamento e retorna o resultado validado como PredictionResult
    """

    raw_df = pd.json_normalize(payload, sep="_")
    engineered = feature_engineering(raw_df)

    engineered = engineered.drop(columns=[c for c in engineered.columns if c == "target"], errors="ignore")

    model, version = load_latest_model()
    prob = model.predict_proba(engineered)[:, 1][0]
    pred = bool(prob >= 0.5)

    result = PredictionResult(
        prediction=pred,
        probability=round(prob, 4),
        model_version=version,
        timestamp=datetime.utcnow(),
    )
    return result