import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.features import FEATURES
from app.schemas import PredictRequest, PredictResponse

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "model" / "model.joblib")
)


class RiskModel:
    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {path}. Ejecuta: python scripts/train_dummy.py"
            )
        self.model = joblib.load(path)
        importancias = getattr(self.model, "feature_importances_", np.ones(len(FEATURES)))
        orden = np.argsort(importancias)[::-1]
        self.ranking = [FEATURES[i] for i in orden]

    def predict(self, reqs: list[PredictRequest]) -> list[PredictResponse]:
        X = pd.DataFrame([[r.features[f] for f in FEATURES] for r in reqs], columns=FEATURES)
        probas = self.model.predict_proba(X)[:, 1]
        return [
            PredictResponse(
                student_id=r.student_id,
                riesgo=self._nivel(p, r.umbral_amarillo, r.umbral_rojo),
                probabilidad=round(float(p), 4),
                factores=self.ranking[:3],
            )
            for r, p in zip(reqs, probas)
        ]

    @staticmethod
    def _nivel(p: float, amarillo: float, rojo: float) -> str:
        if p >= rojo:
            return "rojo"
        if p >= amarillo:
            return "amarillo"
        return "verde"
