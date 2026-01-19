import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from .predict import PredictionResult, predict_from_raw

app = FastAPI(
    title="Probabilidade de Acumular",
    version="0.1.0",
    description="Prediz a chance do próximo jogo acumular",
)

class LotteryRequest(BaseModel):
    loteria: str
    concurso: int
    data: str
    local: str | None = None
    dezenasOrdemSorteio: list[str] | None = None
    dezenas: list[str] | None = None
    trevos: str | None = None
    timeCoracao: str | None = None
    mesSorte: str | None = None
    premiacoes: list[dict] | None = None
    estadosPremiados: list | None = None
    observacao: str | None = None
    acumulou: bool | None = None
    proximoConcurso: int | None = None
    dataProximoConcurso: str | None = None
    localGanhadores: list[dict] | None = None
    valorArrecadado: float | None = None
    valorAcumuladoConcurso_0_5: float | None = None
    valorAcumuladoConcursoEspecial: float | None = None
    valorAcumuladoProximoConcurso: float | None = None
    valorEstimadoProximoConcurso: float | None = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=PredictionResult | list[PredictionResult],
    summary="Prevê se o próximo concurso irá acumular",
    response_description="Predição, probabilidade, versão do modelo, timestamp",
)
async def predict(request: LotteryRequest | list[LotteryRequest]):
    """
    **POST** um registro de sorteio.

    O payload será validado e o modelo será executado.

    Retorno:
    * `prediction` – `true` quando o modelo indicar que o próximo sorteio irá acumular  
    * `probability` – confiança do modelo
    * `model_version` – versão do modelo  
    * `timestamp` – indicação de data e hora em que o resultado foi gerado
    """
    try:
        requests = request if isinstance(request, list) else [request]
        results = [predict_from_raw(item.model_dump()) for item in requests]
        if isinstance(request, list):
            return JSONResponse(content=[item.model_dump(mode="json") for item in results])
        return JSONResponse(content=results[0].model_dump(mode="json"))
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as exc:
        import traceback, sys

        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(exc))
