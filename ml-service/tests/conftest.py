import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def client():
    if not (RAIZ / "model" / "model.joblib").exists():
        subprocess.run([sys.executable, str(RAIZ / "scripts" / "train_dummy.py")], check=True)
    from app.main import app

    with TestClient(app) as c:
        yield c


def alumno(student_id: int = 1, **cambios) -> dict:
    features = {
        "promedio": 15, "inasistencias": 1, "conducta": 1, "horas_estudio": 6,
        "nivel_edu_padres": 3, "acceso_tecnologia": 1, "tiempo_disponible_padres": 3,
        "comunicacion_familiar": 3, "cursos_desaprobados": 0, "tareas_entregadas_pct": 0.95,
    }
    features.update(cambios)
    return {"student_id": student_id, "features": features}
