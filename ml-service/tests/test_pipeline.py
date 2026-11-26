"""Pruebas del pipeline de entrenamiento: datos, limpieza, VIF, umbral y compuerta de calidad."""

import numpy as np
import pandas as pd

from app.features import FEATURES
from pipeline import entrenar as ent
from pipeline.dataset import OBJETIVO, generar


def test_dataset_reproducible_y_completo():
    a, b = generar(300, seed=7), generar(300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert set(FEATURES) | {OBJETIVO} == set(a.columns)
    assert set(a[OBJETIVO].unique()) <= {0, 1}
    assert not generar(300, seed=8).equals(a)


def test_limpiar_marca_fuera_de_rango_como_faltante():
    df = generar(5)
    df.loc[0, "promedio"] = 25
    df.loc[1, "conducta"] = 9
    df["horas_estudio"] = df["horas_estudio"].astype(object)
    df.loc[2, "horas_estudio"] = "no sabe"
    limpio = ent.limpiar(df)
    assert pd.isna(limpio.loc[0, "promedio"])
    assert pd.isna(limpio.loc[1, "conducta"])
    assert pd.isna(limpio.loc[2, "horas_estudio"])
    assert limpio.loc[3, "promedio"] == df.loc[3, "promedio"]


def test_vif_detecta_y_descarta_colinealidad():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    X = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(scale=0.01, size=500), "c": rng.normal(size=500)})
    vif = ent.calcular_vif(X)
    assert vif["a"] > 5 and vif["b"] > 5 and vif["c"] < 2
    seleccion, _ = ent.seleccionar_por_vif(X)
    assert "c" in seleccion and len(seleccion) == 2


def test_metricas_y_matriz_de_confusion():
    y = np.array([1, 1, 0, 0])
    m = ent.metricas(y, np.array([0.9, 0.3, 0.6, 0.1]), 0.5)
    assert m["matriz_confusion"] == {"vn": 1, "fp": 1, "fn": 1, "vp": 1}
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_compuerta_de_calidad_no_publica_si_no_cumple(tmp_path, monkeypatch):
    paquete = {"modelo": None}
    reporte = {
        "version": "rf-x", "variables_modelo": FEATURES, "cumple_rnf02": False,
        "validacion_cruzada": {"f1_media": 0.7, "f1_desviacion": 0.01},
        "prueba": {"f1": 0.7, "precision": 0.7, "recall": 0.7, "auc_roc": 0.85},
    }
    monkeypatch.setattr(ent, "entrenar", lambda: (paquete, reporte))
    salida, rep = tmp_path / "m.joblib", tmp_path / "r.json"
    monkeypatch.setattr("sys.argv", ["entrenar", "--salida", str(salida), "--reporte", str(rep)])
    assert ent.main() == 1
    assert rep.exists() and not salida.exists()

    monkeypatch.setattr("sys.argv", ["entrenar", "--salida", str(salida), "--reporte", str(rep), "--forzar"])
    assert ent.main() == 0
    assert salida.exists()
