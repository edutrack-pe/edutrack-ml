from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_loader import RiskModel
from app.schemas import PredictRequest, PredictResponse

estado: dict[str, RiskModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado["modelo"] = RiskModel()
    yield
    estado.clear()


app = FastAPI(
    title="EduTrack ML Service",
    description="Predicción del riesgo de desaprobación/deserción (Random Forest).",
    version="0.1.0",
    lifespan=lifespan,
)


def _modelo() -> RiskModel:
    if "modelo" not in estado:
        raise HTTPException(503, "Modelo no cargado")
    return estado["modelo"]


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": "modelo" in estado}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _modelo().predict([req])[0]


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(reqs: list[PredictRequest]):
    if not reqs:
        return []
    return _modelo().predict(reqs)
