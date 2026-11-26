import os
from pathlib import Path

import joblib
import pandas as pd

from app.features import FEATURES
from app.schemas import PredictRequest, PredictResponse

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "model" / "model.joblib")
)
# Un factor se reporta si, al llevarlo al valor de referencia, la probabilidad baja al menos este monto.
APORTE_MINIMO = 0.02
MAX_FACTORES = 3


class RiskModel:
    """Carga el paquete generado por `python -m pipeline.entrenar` (pipeline + metadatos)."""

    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {path}. Ejecuta: python -m pipeline.entrenar"
            )
        paquete = joblib.load(path)
        if not isinstance(paquete, dict) or "modelo" not in paquete:
            raise ValueError("Formato de modelo no reconocido: vuelve a entrenar con python -m pipeline.entrenar")
        self.modelo = paquete["modelo"]
        self.version: str = paquete["version"]
        self.variables: list[str] = paquete["variables_modelo"]
        self.referencia: dict[str, float] = paquete["referencia"]
        self.info = {k: paquete[k] for k in ("version", "variables_modelo", "umbral_decision", "metricas",
                                             "fecha_entrenamiento") if k in paquete}

    def _probabilidades(self, filas: pd.DataFrame) -> list[float]:
        return [float(p) for p in self.modelo.predict_proba(filas[self.variables])[:, 1]]

    def factores(self, fila: pd.DataFrame, probabilidad: float) -> list[str]:
        """Explicación individual por oclusión: cuánto baja la probabilidad si una variable toma el
        valor típico de un estudiante sin riesgo. Se devuelven las variables con mayor aporte."""
        alternativas = pd.concat([fila] * len(self.variables), ignore_index=True)
        for i, var in enumerate(self.variables):
            alternativas.loc[i, var] = self.referencia[var]
        aportes = {var: probabilidad - p for var, p in zip(self.variables, self._probabilidades(alternativas))}
        ordenados = sorted((v for v in aportes if aportes[v] >= APORTE_MINIMO), key=aportes.get, reverse=True)
        return ordenados[:MAX_FACTORES]

    def predict(self, reqs: list[PredictRequest]) -> list[PredictResponse]:
        X = pd.DataFrame([[r.features[f] for f in FEATURES] for r in reqs], columns=FEATURES)
        probas = self._probabilidades(X)
        respuestas = []
        for i, (r, p) in enumerate(zip(reqs, probas)):
            riesgo = self._nivel(p, r.umbral_amarillo, r.umbral_rojo)
            factores = self.factores(X.iloc[[i]].reset_index(drop=True), p) if riesgo != "verde" else []
            respuestas.append(PredictResponse(
                student_id=r.student_id, riesgo=riesgo, probabilidad=round(p, 4),
                factores=factores, modelo_version=self.version,
            ))
        return respuestas

    @staticmethod
    def _nivel(p: float, amarillo: float, rojo: float) -> str:
        if p >= rojo:
            return "rojo"
        if p >= amarillo:
            return "amarillo"
        return "verde"
