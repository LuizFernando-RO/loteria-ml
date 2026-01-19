import json
import pathlib
from datetime import datetime
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def load_json_dataset(path: pathlib.Path) -> pd.DataFrame:
    """
    Lê o arquivo `dataset.json` contendo um JSON array e retorna um 
    dataset sanitizado
    """
    logger.info(f"Loading raw data from {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    
    raw = [rec for rec in raw if rec is not None]

    df = pd.json_normalize(raw, sep="_")
    logger.debug(f"Raw shape after dropping nulls: {df.shape}")
    return df

def parse_date(date_str: str) -> pd.Timestamp | None:
    """Parser para data. Em caso de falha, retorna NaT"""
    try:
        return pd.to_datetime(date_str, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

def int_list(lst):
    return [int(v) for v in lst] if isinstance(lst, list) else []

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma os dados de entrada, selecionando e adicionando
    features relevantes
    """

    logger.info("Running feature engineering")

    df["date"] = df["data"].apply(parse_date)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df["n_dezenas"] = df["dezenas"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

    df["sum_dezenas"] = df["dezenas"].apply(lambda x: sum(int_list(x)))
    df["mean_dezenas"] = df["dezenas"].apply(lambda x: np.mean(int_list(x)) if x else np.nan)

    monetary_cols = [
        "valorArrecadado",
        "valorAcumuladoConcurso_0_5",
        "valorAcumuladoConcursoEspecial",
        "valorAcumuladoProximoConcurso",
        "valorEstimadoProximoConcurso",
    ]

    for col in monetary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    drop_cols = [
        "loteria", "concurso", "data", "local", "dezenasOrdemSorteio",
        "dezenas", "trevos", "timeCoracao", "mesSorte", "premiacoes",
        "estadosPremiados", "observacao", "proximoConcurso", "dataProximoConcurso",
        "localGanhadores",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.debug(f"Engineered shape: {df.shape}")
    return df

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    """
    
    if "concurso" in df.columns:
        df = df.sort_values("concurso")
    elif "date" in df.columns:
        df = df.sort_values("date")
    else:
        logger.warning("No obvious ordering column – using original order")

    def _to_acumulou_int(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
            return int(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "sim", "yes", "y"):
                return 1
            if normalized in ("false", "0", "nao", "não", "no", "n"):
                return 0
        return np.nan

    df["acumulou_int"] = df["acumulou"].apply(_to_acumulou_int)
    df = df.dropna(subset=["acumulou_int"]).reset_index(drop=True)

    df["target"] = df["acumulou_int"].shift(-1)

    df = df.dropna(subset=["target"]).reset_index(drop=True)

    X = df.drop(columns=["target", "acumulou_int"])
    y = df["target"].astype(int)
    return X, y

def build_training_pipeline() -> Pipeline:
    """
    """
    logger.info("Building training pipeline")

    numeric_features = [
        "year", "month", "day_of_week",
        "n_dezenas", "sum_dezenas", "mean_dezenas",
        "valorArrecadado",
        "valorAcumuladoConcurso_0_5",
        "valorAcumuladoConcursoEspecial",
        "valorAcumuladoProximoConcurso",
        "valorEstimadoProximoConcurso",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model),
        ]
    )
    return pipeline

def train_and_save(
    raw_data_path: pathlib.Path,
    model_dir: pathlib.Path,
    version: str | None = None,
) -> Tuple[Pipeline, str]:
    """
    """
    df_raw = load_json_dataset(raw_data_path)
    df_feat = feature_engineering(df_raw)

    X, y = prepare_training_data(df_feat)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    logger.info(f"ROC‑AUC: {auc:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    if not version:
        version = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir / f"model_{version}.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Modelo salvo em {model_path}")

    meta = {"version": version, "created_at": datetime.utcnow().isoformat() + "Z", "auc": auc}
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return pipeline, version
