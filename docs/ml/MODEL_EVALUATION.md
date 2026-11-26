# Evaluación del modelo · rf-20261126-9ad6a250

Fuente: `ml-service/reports/metricas.json`, generado por `python -m pipeline.entrenar` el 26/11/2026. Ningún valor se ajustó a mano y el conjunto de prueba no se modificó.

## Protocolo
1. Dataset sintético de 2 000 filas (`pipeline/dataset.py`, semilla 42); 28,3 % en riesgo.
2. Limpieza (rangos válidos, imputación por mediana dentro del pipeline).
3. División estratificada 80/20 → **1 600 entrenamiento / 400 prueba**. La prueba se usa **una sola vez** al final.
4. VIF sobre el entrenamiento: se descartaría toda variable con VIF > 5. Máximo observado 1,617 (`promedio`, `cursos_desaprobados`) → **ninguna descartada**.
5. `Pipeline(SimpleImputer → SMOTE → RandomForest)`: SMOTE se aplica solo dentro de cada *fold* de entrenamiento.
6. `GridSearchCV` con `StratifiedKFold(5)`, optimizando F1. Mejores hiperparámetros: `n_estimators=400`, `max_depth=6`, `min_samples_leaf=3`.
7. Umbral de decisión elegido con predicciones fuera de *fold* del **entrenamiento** (rango 0,30–0,65): **0,50**.

## Resultados
### Validación cruzada (entrenamiento, 5 folds)
| Métrica | Valor |
|---|---|
| F1 | 0,7646 ± 0,018 |
| AUC‑ROC | 0,9144 |

### Conjunto de prueba (400 estudiantes, umbral 0,50)
| Métrica | Valor | Meta RNF‑02 | Estado |
|---|---|---|---|
| Precisión | 0,6842 | — | — |
| Recall | 0,8053 | — | — |
| **F1** | **0,7398** | ≥ 0,75 | ❌ |
| **AUC‑ROC** | **0,9122** | > 0,80 | ✅ |

Matriz de confusión: VN 245 · FP 42 · FN 22 · VP 91.

### Sensibilidad al umbral (mismo conjunto de prueba, solo informativo)
| Umbral | Precisión | Recall | F1 | Uso |
|---|---|---|---|---|
| 0,40 | 0,6533 | 0,8673 | 0,7452 | Límite verde/amarillo del sistema |
| 0,50 | 0,6842 | 0,8053 | 0,7398 | Umbral de decisión del modelo |
| 0,70 | 0,7802 | 0,6283 | 0,6961 | Límite de riesgo alto (alertas) |

> No se eligió el umbral mirando la prueba: hacerlo para pasar de 0,740 a 0,745 sería una fuga de información y, aun así, no alcanzaría 0,75.

### Importancia de variables (Gini)
promedio 0,459 · cursos_desaprobados 0,305 · inasistencias 0,052 · tareas_entregadas_pct 0,038 · nivel_edu_padres 0,035 · conducta 0,030 · comunicacion_familiar 0,030 · horas_estudio 0,029 · tiempo_disponible_padres 0,013 · acceso_tecnologia 0,009.

## Conclusión
- **AUC 0,912**: el modelo ordena bien a los estudiantes por riesgo.
- **F1 0,740 < 0,75: RNF‑02 no se cumple.** La CV (0,765) supera la meta, pero la prueba independiente no; se reporta el valor de prueba.
- El *quality gate* (`entrenar.py` devuelve código 1 si no cumple) funcionó. El modelo usado en la demo se publicó con `--forzar`, decisión documentada como O‑01.
- Causa probable: el dataset sintético incluye un sorteo de Bernoulli (ruido irreducible) y un techo de desempeño cercano a la meta.

## Próximos pasos
1. Entrenar con datos reales anonimizados y con consentimiento (Ley N° 29733).
2. Añadir variables temporales (tendencia de notas entre bimestres, asistencia por mes).
3. Probar *gradient boosting* y calibración de probabilidades, manteniendo la misma partición de prueba.
