from tests.conftest import alumno


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_cargado"] is True


def test_predict_contrato(client):
    r = client.post("/predict", json=alumno())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"student_id", "riesgo", "probabilidad", "factores"}
    assert body["riesgo"] in {"verde", "amarillo", "rojo"}
    assert 0 <= body["probabilidad"] <= 1
    assert len(body["factores"]) == 3


def test_alumno_bueno_vs_alumno_en_riesgo(client):
    bueno = client.post("/predict", json=alumno()).json()
    malo = client.post("/predict", json=alumno(
        promedio=7, inasistencias=15, conducta=3, horas_estudio=0, comunicacion_familiar=1,
        cursos_desaprobados=6, tareas_entregadas_pct=0.1,
    )).json()
    assert bueno["riesgo"] == "verde"
    assert malo["riesgo"] == "rojo"
    assert malo["probabilidad"] > bueno["probabilidad"]


def test_umbrales_personalizados(client):
    req = alumno() | {"umbral_amarillo": 0.0001, "umbral_rojo": 0.0002}
    assert client.post("/predict", json=req).json()["riesgo"] == "rojo"


def test_batch(client):
    r = client.post("/predict/batch", json=[alumno(1), alumno(2)])
    assert r.status_code == 200
    assert [p["student_id"] for p in r.json()] == [1, 2]
    assert client.post("/predict/batch", json=[]).json() == []


def test_faltan_variables(client):
    r = client.post("/predict", json={"student_id": 1, "features": {"promedio": 12}})
    assert r.status_code == 422


def test_umbrales_invalidos(client):
    req = alumno() | {"umbral_amarillo": 0.8, "umbral_rojo": 0.5}
    assert client.post("/predict", json=req).status_code == 422
