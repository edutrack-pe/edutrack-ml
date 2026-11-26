"""Dataset SINTÉTICO y reproducible para entrenar el modelo de riesgo.

No existe un dataset autorizado de estudiantes reales (Ley N° 29733), por eso el modelo se entrena con
datos ficticios. El proceso generador es el mismo que el proyecto usaba desde su primera versión
(`scripts/train_dummy.py`); no se ajustó para alcanzar las metas de RNF-02.

Variable objetivo `en_riesgo` (0/1): el estudiante desaprueba el año o abandona el colegio.
Se simula con un puntaje latente `z` (combinación lineal de las variables) y un sorteo de Bernoulli con
probabilidad sigmoide(z). Ese sorteo introduce ruido realista: dos estudiantes con el mismo perfil
pueden tener desenlaces distintos.
"""

import numpy as np
import pandas as pd

OBJETIVO = "en_riesgo"


def generar(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "promedio": rng.normal(13, 2.5, n).clip(0, 20),
        "inasistencias": rng.poisson(3, n),
        "conducta": rng.choice([1, 2, 3], n, p=[0.6, 0.3, 0.1]),
        "horas_estudio": rng.gamma(2, 2, n).clip(0, 20),
        "nivel_edu_padres": rng.choice([1, 2, 3], n, p=[0.2, 0.5, 0.3]),
        "acceso_tecnologia": rng.choice([0, 1], n, p=[0.25, 0.75]),
        "tiempo_disponible_padres": rng.choice([1, 2, 3], n),
        "comunicacion_familiar": rng.choice([1, 2, 3], n, p=[0.25, 0.45, 0.3]),
        "tareas_entregadas_pct": rng.beta(5, 2, n),
    })
    df["cursos_desaprobados"] = rng.poisson(np.clip((14 - df["promedio"]) / 2, 0.1, None))
    z = (
        -0.9 * (df["promedio"] - 13)
        + 0.35 * df["inasistencias"]
        + 0.6 * (df["conducta"] - 1)
        - 0.15 * df["horas_estudio"]
        - 0.4 * (df["comunicacion_familiar"] - 2)
        - 0.3 * (df["nivel_edu_padres"] - 2)
        + 0.5 * df["cursos_desaprobados"]
        - 2.5 * (df["tareas_entregadas_pct"] - 0.7)
        - 3.0
    )
    df[OBJETIVO] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df
