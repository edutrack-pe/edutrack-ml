from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.features import FEATURES

Riesgo = Literal["verde", "amarillo", "rojo"]


class PredictRequest(BaseModel):
    student_id: int
    features: dict[str, float]
    umbral_amarillo: float = Field(0.4, ge=0, le=1)
    umbral_rojo: float = Field(0.7, ge=0, le=1)

    @model_validator(mode="after")
    def validar(self) -> "PredictRequest":
        faltantes = [f for f in FEATURES if f not in self.features]
        if faltantes:
            raise ValueError(f"Faltan variables: {', '.join(faltantes)}")
        if self.umbral_amarillo >= self.umbral_rojo:
            raise ValueError("umbral_amarillo debe ser menor que umbral_rojo")
        return self


class PredictResponse(BaseModel):
    student_id: int
    riesgo: Riesgo
    probabilidad: float
    factores: list[str]
    modelo_version: str
