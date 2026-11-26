"""Entrena, evalúa y publica el modelo de riesgo (HU04 / RNF-02).

Uso:  python -m pipeline.entrenar [--forzar] [--salida model/model.joblib]

Pasos:
 1. Dataset sintético reproducible (pipeline/dataset.py).
 2. Limpieza: rangos válidos y valores faltantes (imputación por mediana dentro del pipeline).
 3. Variables categóricas: ya son ordinales numéricas (1-3) o binarias (0/1); no requieren one-hot.
 4. División estratificada 80 % entrenamiento / 20 % prueba ANTES de cualquier ajuste.
 5. VIF calculado solo con el conjunto de entrenamiento; se descartan variables con VIF > 5.
 6. SMOTE dentro del pipeline de imbalanced-learn: se aplica solo a los folds de entrenamiento.
 7. Random Forest con búsqueda de hiperparámetros y validación cruzada estratificada de 5 folds (F1).
 8. Evaluación única sobre el 20 % de prueba: matriz de confusión, precision, recall, F1 y AUC-ROC.
 9. Compuerta de calidad: si F1 < 0.75 o AUC-ROC <= 0.80 no se publica el modelo (salvo --forzar).
10. Serialización del pipeline con versión, fecha, variables y métricas.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from app.features import FEATURES  # noqa: E402
from pipeline.dataset import OBJETIVO, generar  # noqa: E402

SEMILLA = 42
UMBRAL_VIF = 5.0
# "En riesgo" = riesgo moderado o alto. El límite inferior del moderado se elige por validación cruzada
# (elegir_umbral); el límite del riesgo alto es el documentado en el TB4.
UMBRAL_ALTO = 0.70
META_F1, META_AUC = 0.75, 0.80
RANGOS = {
    "promedio": (0, 20), "inasistencias": (0, None), "conducta": (1, 3), "horas_estudio": (0, 60),
    "nivel_edu_padres": (1, 3), "acceso_tecnologia": (0, 1), "tiempo_disponible_padres": (1, 3),
    "comunicacion_familiar": (1, 3), "cursos_desaprobados": (0, None), "tareas_entregadas_pct": (0, 1),
}
REJILLA = {
    "rf__n_estimators": [200, 400],
    "rf__max_depth": [6, 10, None],
    "rf__min_samples_leaf": [1, 3, 5],
}


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Lleva cada variable a su rango válido; los valores fuera de rango o no numéricos quedan como faltantes."""
    df = df.copy()
    for col, (minimo, maximo) in RANGOS.items():
        valores = pd.to_numeric(df[col], errors="coerce")
        fuera = (valores < minimo) | ((valores > maximo) if maximo is not None else False)
        df[col] = valores.mask(fuera)
    return df.dropna(subset=[OBJETIVO])


def calcular_vif(X: pd.DataFrame) -> dict[str, float]:
    """VIF_i = 1 / (1 - R²_i), con R²_i de la regresión lineal de la variable i sobre las demás."""
    datos = X.fillna(X.median()).to_numpy(dtype=float)
    vif = {}
    for i, col in enumerate(X.columns):
        y = datos[:, i]
        otras = np.column_stack([np.ones(len(datos)), np.delete(datos, i, axis=1)])
        coef, *_ = np.linalg.lstsq(otras, y, rcond=None)
        residuo = y - otras @ coef
        r2 = 1 - residuo.var() / y.var() if y.var() > 0 else 0.0
        vif[col] = float(np.inf if r2 >= 1 else 1 / (1 - r2))
    return vif


def seleccionar_por_vif(X: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Elimina de una en una la variable con mayor VIF mientras supere el umbral."""
    variables = list(X.columns)
    while True:
        vif = calcular_vif(X[variables])
        peor = max(vif, key=vif.get)
        if vif[peor] <= UMBRAL_VIF or len(variables) <= 2:
            return variables, vif
        variables.remove(peor)


def metricas(y: np.ndarray, proba: np.ndarray, umbral: float) -> dict:
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "umbral": umbral,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "matriz_confusion": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
    }


def elegir_umbral(estimador, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> float:
    """Umbral de decisión que maximiza el F1 con predicciones fuera de fold del ENTRENAMIENTO.

    El conjunto de prueba no participa: solo se usa después, una vez, para medir.
    Se limita a [0.30, 0.65] para que "en riesgo" (moderado o alto) quede por debajo del umbral alto de 0.70.
    """
    oof = cross_val_predict(clone(estimador), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    candidatos = np.round(np.arange(0.30, 0.651, 0.01), 2)
    f1s = [f1_score(y, (oof >= u).astype(int)) for u in candidatos]
    return float(candidatos[int(np.argmax(f1s))])


def entrenar() -> tuple[dict, dict]:
    df = limpiar(generar())
    X, y = df[FEATURES], df[OBJETIVO].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)

    seleccion, vif = seleccionar_por_vif(X_tr)

    pipe = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=SEMILLA)),
        ("rf", RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(pipe, REJILLA, scoring={"f1": "f1", "auc": "roc_auc"}, refit="f1", cv=cv, n_jobs=-1)
    busqueda.fit(X_tr[seleccion], y_tr)
    modelo = busqueda.best_estimator_
    mejor = busqueda.best_index_
    resultados_cv = busqueda.cv_results_

    umbral = elegir_umbral(busqueda.best_estimator_, X_tr[seleccion], y_tr, cv)
    proba = modelo.predict_proba(X_te[seleccion])[:, 1]
    principal = metricas(y_te.to_numpy(), proba, umbral)
    auc = round(float(roc_auc_score(y_te, proba)), 4)
    importancias = dict(sorted(
        zip(seleccion, (round(float(v), 4) for v in modelo.named_steps["rf"].feature_importances_)),
        key=lambda kv: kv[1], reverse=True,
    ))
    huella = hashlib.sha256(json.dumps(
        {"params": busqueda.best_params_, "variables": seleccion, "umbral": umbral}, sort_keys=True, default=str,
    ).encode()).hexdigest()[:8]
    version = f"rf-{datetime.now(timezone.utc):%Y%m%d}-{huella}"

    reporte = {
        "version": version,
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos": {
            "origen": "sintético (pipeline/dataset.py, semilla 42)",
            "filas": int(len(df)), "entrenamiento": int(len(X_tr)), "prueba": int(len(X_te)),
            "proporcion_en_riesgo": round(float(y.mean()), 4),
        },
        "vif": {k: round(v, 3) for k, v in vif.items()},
        "variables_descartadas_vif": [f for f in FEATURES if f not in seleccion],
        "variables_modelo": seleccion,
        "mejores_hiperparametros": {k.removeprefix("rf__"): v for k, v in busqueda.best_params_.items()},
        "umbral_decision": {"valor": umbral, "metodo": "máximo F1 fuera de fold en entrenamiento (5 folds)"},
        "validacion_cruzada": {
            "folds": 5,
            "f1_media": round(float(resultados_cv["mean_test_f1"][mejor]), 4),
            "f1_desviacion": round(float(resultados_cv["std_test_f1"][mejor]), 4),
            "auc_media": round(float(resultados_cv["mean_test_auc"][mejor]), 4),
        },
        "prueba": {
            **principal,
            "auc_roc": auc,
            "con_umbral_0_5": metricas(y_te.to_numpy(), proba, 0.5),
            "con_umbral_sistema_0_4": metricas(y_te.to_numpy(), proba, 0.4),
            "con_umbral_alto_0_7": metricas(y_te.to_numpy(), proba, UMBRAL_ALTO),
        },
        "importancia_variables": importancias,
        "metas": {"f1_minimo": META_F1, "auc_minimo_exclusivo": META_AUC},
    }
    reporte["cumple_rnf02"] = principal["f1"] >= META_F1 and auc > META_AUC

    # Perfil de referencia para explicar cada predicción: medianas de los estudiantes sin riesgo (entrenamiento).
    referencia = X_tr.loc[y_tr == 0, seleccion].median().to_dict()
    paquete = {
        "modelo": modelo,
        "version": version,
        "variables_entrada": FEATURES,
        "variables_modelo": seleccion,
        "referencia": {k: float(v) for k, v in referencia.items()},
        "umbral_decision": umbral,
        "umbral_alto": UMBRAL_ALTO,
        "metricas": {"f1": principal["f1"], "auc_roc": auc},
        "fecha_entrenamiento": reporte["fecha_entrenamiento"],
    }
    return paquete, reporte


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", default=str(RAIZ / "model" / "model.joblib"))
    parser.add_argument("--reporte", default=str(RAIZ / "reports" / "metricas.json"))
    parser.add_argument("--forzar", action="store_true", help="Publica el modelo aunque no cumpla RNF-02")
    args = parser.parse_args()

    paquete, reporte = entrenar()
    Path(args.reporte).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reporte).write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    p = reporte["prueba"]
    print(f"Versión {reporte['version']} · variables: {len(reporte['variables_modelo'])}")
    print(f"CV F1 {reporte['validacion_cruzada']['f1_media']} ± {reporte['validacion_cruzada']['f1_desviacion']}")
    print(f"Prueba: F1 {p['f1']} · precision {p['precision']} · recall {p['recall']} · AUC-ROC {p['auc_roc']}")

    if not reporte["cumple_rnf02"] and not args.forzar:
        print(f"Compuerta de calidad: F1 >= {META_F1} y AUC-ROC > {META_AUC} no se cumplen. "
              "El modelo NO se publica (RNF-02). Revise el reporte o use --forzar.")
        return 1
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    # La evaluación viaja con el modelo: /modelo la expone sin depender del archivo de reporte.
    joblib.dump({**paquete, "evaluacion": reporte}, args.salida)
    print(f"Modelo publicado en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
