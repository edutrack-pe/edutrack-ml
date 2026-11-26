# Model card · Riesgo académico EduTrack

| Campo | Valor |
|---|---|
| Versión | `rf-20261126-9ad6a250` (formato `rf-AAAAMMDD-hash`, se guarda en cada predicción) |
| Tipo | Clasificación binaria · `RandomForestClassifier` (400 árboles, profundidad 6, hoja mínima 3) dentro de un pipeline imblearn con imputación por mediana y SMOTE |
| Entrenamiento | `python -m pipeline.entrenar` → `model/model.joblib` (no versionado) + `reports/metricas.json` |
| Fecha | 26/11/2026 |
| Datos | 2 000 registros **sintéticos** (ver `DATASET_DESCRIPTION.md`) |

## Uso previsto
Señal **preventiva** para que directores y tutores de secundaria prioricen el acompañamiento de estudiantes con posible riesgo de desaprobación o abandono. Salidas: probabilidad, nivel (verde < 0,40 ≤ amarillo < 0,70 ≤ rojo, configurable) y los factores individuales que más elevan el riesgo.

## Usos no previstos
- Decisiones disciplinarias, de matrícula, de promoción o cualquier sanción.
- Etiquetar a estudiantes ante otros estudiantes o familias. A las familias **no** se les muestra la probabilidad, solo el nivel y recomendaciones.
- Uso con datos reales sin reentrenar, validar y contar con consentimiento.

## Métricas (prueba independiente, 400 filas)
| F1 | AUC‑ROC | Precisión | Recall |
|---|---|---|---|
| 0,740 | 0,912 | 0,684 | 0,805 |

RNF‑02 exige F1 ≥ 0,75 y AUC > 0,80: **AUC cumple, F1 no**. El *quality gate* rechaza el modelo; se publicó con `--forzar` solo para la demo académica. Detalle en `MODEL_EVALUATION.md`.

## Explicabilidad
Para cada estudiante, cada variable se reemplaza por el valor mediano de los estudiantes **sin riesgo** del entrenamiento; los factores son las variables cuya sustitución más reduce la probabilidad (oclusión). Se calcula en lote con una sola llamada a `predict_proba`.

## Consideraciones éticas y riesgos
- **Falsos negativos** (22 de 113 positivos en prueba): estudiantes en riesgo marcados como bajo. El tutor no debe descartar a un estudiante solo por el semáforo.
- **Falsos positivos** (42): posible estigmatización; por eso el lenguaje de las pautas evita términos negativos (verificado por prueba) y la familia ve un aviso de «señal preventiva, no un diagnóstico».
- Variables socioeconómicas (`nivel_edu_padres`, `acceso_tecnologia`) pueden introducir sesgos; con datos reales se debe auditar el desempeño por subgrupo.
- La probabilidad no está calibrada.

## Mantenimiento
Reentrenar cuando existan datos reales; el reentrenamiento automático (MLOps) está fuera del alcance del MVP.
