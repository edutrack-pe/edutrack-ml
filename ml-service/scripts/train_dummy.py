"""Entrena un Random Forest con datos SINTÉTICOS para desbloquear la integración.

Reemplazar por el modelo real exportado desde ml-research/ cuando alcance
F1 >= 0.75 y AUC-ROC > 0.80.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from app.features import FEATURES  # noqa: E402


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
    df["en_riesgo"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def main() -> None:
    df = generar()
    X, y = df[FEATURES], df["en_riesgo"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    modelo = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    print(f"Clase positiva: {y.mean():.1%}")
    print(f"F1 (sintético): {f1_score(y_te, pred):.3f}")
    print(f"AUC-ROC (sintético): {roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1]):.3f}")
    destino = RAIZ / "model" / "model.joblib"
    destino.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, destino)
    print(f"Modelo guardado en {destino}")


if __name__ == "__main__":
    main()
