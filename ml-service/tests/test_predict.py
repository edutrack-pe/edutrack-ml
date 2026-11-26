from tests.conftest import alumno

EN_RIESGO = dict(
    promedio=7, inasistencias=15, conducta=3, horas_estudio=0, comunicacion_familiar=1,
    cursos_desaprobados=6, tareas_entregadas_pct=0.1,
)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["modelo_cargado"] is True
    assert body["modelo_version"].startswith("rf-")


def test_info_del_modelo(client):
    info = client.get("/modelo").json()
    assert set(info) >= {"version", "variables_modelo", "umbral_decision", "metricas", "fecha_entrenamiento"}
    assert 0 <= info["metricas"]["auc_roc"] <= 1
    assert len(info["variables_modelo"]) >= 2


def test_predict_contrato(client):
    r = client.post("/predict", json=alumno())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"student_id", "riesgo", "probabilidad", "factores", "modelo_version"}
    assert body["riesgo"] in {"verde", "amarillo", "rojo"}
    assert 0 <= body["probabilidad"] <= 1


def test_alumno_bueno_vs_alumno_en_riesgo(client):
    bueno = client.post("/predict", json=alumno()).json()
    malo = client.post("/predict", json=alumno(**EN_RIESGO)).json()
    assert bueno["riesgo"] == "verde"
    assert malo["riesgo"] == "rojo"
    assert malo["probabilidad"] > bueno["probabilidad"]


def test_factores_individuales(client):
    """Los factores dependen del perfil de cada estudiante, no son un ranking global fijo."""
    bueno = client.post("/predict", json=alumno()).json()
    assert bueno["factores"] == []  # en riesgo bajo no se señalan factores

    solo_faltas = client.post("/predict", json=alumno(
        promedio=9, inasistencias=18, cursos_desaprobados=4, tareas_entregadas_pct=0.95)).json()
    solo_tareas = client.post("/predict", json=alumno(
        promedio=9, inasistencias=0, cursos_desaprobados=4, tareas_entregadas_pct=0.05)).json()
    assert "tareas_entregadas_pct" in solo_tareas["factores"]
    assert "tareas_entregadas_pct" not in solo_faltas["factores"]
    assert solo_faltas["factores"] != solo_tareas["factores"]
    assert 1 <= len(solo_faltas["factores"]) <= 3


def test_umbrales_personalizados(client):
    req = alumno() | {"umbral_amarillo": 0.0001, "umbral_rojo": 0.0002}
    assert client.post("/predict", json=req).json()["riesgo"] == "rojo"


def test_batch(client):
    r = client.post("/predict/batch", json=[alumno(1), alumno(2, **EN_RIESGO)])
    assert r.status_code == 200
    assert [p["student_id"] for p in r.json()] == [1, 2]
    assert client.post("/predict/batch", json=[]).json() == []


def test_es_deterministico(client):
    a = client.post("/predict", json=alumno(**EN_RIESGO)).json()
    b = client.post("/predict", json=alumno(**EN_RIESGO)).json()
    assert a == b


def test_faltan_variables(client):
    r = client.post("/predict", json={"student_id": 1, "features": {"promedio": 12}})
    assert r.status_code == 422


def test_umbrales_invalidos(client):
    req = alumno() | {"umbral_amarillo": 0.8, "umbral_rojo": 0.5}
    assert client.post("/predict", json=req).status_code == 422
