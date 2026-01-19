# Desafio - Previsão de Acúmulo

Serviço de inferência para prever se o próximo concurso vai acumular (sem ganhador).
O modelo usa um conjunto de features simples derivadas dos dados de concursos e
expõe um endpoint HTTP para predição.

## Arquitetura

- `data/raw/dataset.json`: dataset bruto (JSON array)
- `app/pipeline.py`: ingestão, limpeza, feature engineering, treino e persistência.
- `app/predict.py`: carrega o modelo e executa inferência.
- `app/main.py`: API construída utilizando FastAPI com validação do payload e retorno estruturado.
- `app/model/`: artefatos do modelo (`model_*.pkl`) e `metadata.json`.

## Requisitos

- Python 3.11+
- Dependencias em `requirements.txt`

## Setup local

1) Criar ambiente e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Treinar modelo (gera `app/model/model_*.pkl` e `app/model/metadata.json`):

```bash
python - <<'PY'
from pathlib import Path
from app.pipeline import train_and_save

data_path = Path("data/raw/dataset.json")
model_dir = Path("app/model")
train_and_save(data_path, model_dir)
PY
```

3) Subir API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4) Testar:

```bash
pytest -q
```

## Setup via Docker

Build:

```bash
docker build -t loteria-ml .
```

Run:

```bash
docker run --rm -p 8000:8000 loteria-ml
```

## Endpoint

`POST /predict`

Entrada: um JSON no mesmo formato do dataset

```json
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loteria":"MEGA-SENA",
    "concurso":2750,
    "data":"13/04/2024",
    "local":"Sao Paulo, SP",
    "dezenasOrdemSorteio":["04","11","19","23","37","56"],
    "dezenas":["04","11","19","23","37","56"],
    "trevos":null,
    "timeCoracao":null,
    "mesSorte":null,
    "premiacoes":[],
    "estadosPremiados":[],
    "observacao":null,
    "acumulou":true,
    "proximoConcurso":2751,
    "dataProximoConcurso":"16/04/2024",
    "localGanhadores":[],
    "valorArrecadado":52345678.9,
    "valorAcumuladoConcurso_0_5":0.0,
    "valorAcumuladoConcursoEspecial":0.0,
    "valorAcumuladoProximoConcurso":12000000.0,
    "valorEstimadoProximoConcurso":15000000.0
  }'

```

Saída:

```json
{
  "prediction": true,
  "probability": 0.7321,
  "model_version": "20260117T213003Z",
  "timestamp": "2026-01-17T21:30:03.123456"
}
```

## Decisões técnicas

- Alvo: probabilidade de acumulação do próximo concurso.
- Feature engineering simples com data, estatistícas das dezenas e valores monetários.
- Modelo: `GradientBoostingClassifier` por ser leve, rapido e explicavel.
- Pipeline sklearn com imputação e padronização para robustez a valores faltantes.
- Validacao de payload via Pydantic no FastAPI.

## Reprodutibilidade

O treino é determinístico com `random_state=42` (arquivo app/pipeline.py linha 217). Para reprocessar do zero:

1) Garanta o dataset em `data/raw/dataset.json`.
2) Execute o trecho de treino em "Setup local".
3) A API usa o arquivo mais recente em `app/model/`.

## Monitoramento (proposta)

- Log de entrada e saida
- Não há dados sensíveis, logo, não há necessidade de criptografia ou restrição de acesso
- Métricas de latência (tempo de resposta da requisição), taxa de erro e distribuicao dos dados de entrada e/ou saída do modelo
- Uso de ferramentas para tracing das requisições da API, possibilitando a captura de erros em diferentes níveis do sistema
- Alerta para drift de features (ex.: mudanca na distribuicao de dezenas).
- Re-treino periodico ou acionado por drift.