# Evaluación del modelo · rf-20261127-7734085c

Fuente: `ml-service/reports/metricas.json`, generado por `python -m pipeline.entrenar` el 27/11/2026 (UTC). Ningún valor se ajustó a mano y no se eligió nada mirando el conjunto de prueba.

## Protocolo
1. Dataset sintético de **10 000 filas** (`pipeline/dataset.py`, semilla 42); 28,5 % en riesgo.
2. Limpieza (rangos válidos, imputación por mediana dentro del pipeline).
3. División estratificada 80/20 → **8 000 entrenamiento / 2 000 prueba**. La prueba se usa **una sola vez** al final.
4. VIF sobre el entrenamiento: se descartaría toda variable con VIF > 5. Máximo observado 1,642 (`promedio`) → **ninguna descartada**.
5. `Pipeline(SimpleImputer → SMOTE → RandomForest)`: SMOTE se aplica solo dentro de cada *fold* de entrenamiento.
6. `GridSearchCV` con `StratifiedKFold(5)`, optimizando F1 (misma rejilla que antes). Mejores hiperparámetros: `n_estimators=400`, `max_depth=None`, `min_samples_leaf=5`.
7. Umbral de decisión elegido con predicciones fuera de *fold* del **entrenamiento** (rango 0,30–0,65): **0,49**.

## Resultados
### Validación cruzada (entrenamiento, 5 folds)
| Métrica | Valor |
|---|---|
| F1 | 0,7482 ± 0,0193 |
| AUC‑ROC | 0,9124 |

### Conjunto de prueba (2 000 estudiantes, umbral 0,49)
| Métrica | Valor | Meta RNF‑02 | Estado |
|---|---|---|---|
| Precisión | 0,7359 | — | — |
| Recall | 0,7758 | — | — |
| **F1** | **0,7553** | ≥ 0,75 | ✅ |
| **AUC‑ROC** | **0,9229** | > 0,80 | ✅ |

Matriz de confusión: VN 1 270 · FP 159 · FN 128 · VP 443.

### Sensibilidad al umbral (mismo conjunto de prueba, solo informativo)
| Umbral | Precisión | Recall | F1 | Uso |
|---|---|---|---|---|
| 0,40 | 0,6957 | 0,8406 | 0,7613 | Límite verde/amarillo del sistema |
| 0,49 | 0,7359 | 0,7758 | 0,7553 | Umbral de decisión del modelo |
| 0,50 | 0,7436 | 0,7671 | 0,7552 | Referencia |
| 0,70 | 0,8341 | 0,6077 | 0,7031 | Límite de riesgo alto (alertas) |

### Importancia de variables (Gini)
promedio 0,456 · cursos_desaprobados 0,234 · horas_estudio 0,063 · tareas_entregadas_pct 0,063 · inasistencias 0,055 · conducta 0,045 · comunicacion_familiar 0,028 · nivel_edu_padres 0,024 · tiempo_disponible_padres 0,018 · acceso_tecnologia 0,013.

## Conclusión
- **RNF‑02 se cumple en la prueba independiente**: F1 0,755 ≥ 0,75 y AUC 0,923 > 0,80. El *quality gate* aprobó el modelo sin `--forzar`.
- **El margen es pequeño.** La validación cruzada da F1 0,748 ± 0,019, apenas por debajo de la meta: con otra partición el F1 podría quedar un poco bajo 0,75. Se reporta el valor de prueba, como exige el protocolo, sin ocultar el de CV.
- El techo teórico de este dataset (usando la probabilidad real del generador) es F1 ≈ 0,77: el sorteo de Bernoulli introduce ruido que ningún modelo puede eliminar.

## Historial
| Versión | Filas | F1 prueba | AUC prueba | RNF‑02 |
|---|---|---|---|---|
| `rf-20261126-9ad6a250` | 2 000 | 0,740 | 0,912 | ❌ (publicado con `--forzar`) |
| `rf-20261127-7734085c` | 10 000 | 0,755 | 0,923 | ✅ |

**Por qué se cambió el tamaño.** Con 2 000 filas el Random Forest no aprendía bien una frontera suave. Un diagnóstico previo (solo con validación cruzada) mostró que una regresión logística llegaba a 0,767 en este mismo proceso y el techo era 0,77, así que el límite era la cantidad de datos, no el generador. El tamaño de 10 000 se fijó **antes** de reentrenar, se entrenó **una sola vez** y no se volvió a ajustar según el resultado. El generador, la semilla, la rejilla y la proporción 80/20 no cambiaron. Al crecer el dataset también cambió el conjunto de prueba (ahora 2 000 filas), cambio que autorizó el equipo.

## Próximos pasos
1. Entrenar con datos reales anonimizados y con consentimiento (Ley N° 29733).
2. Añadir variables temporales (tendencia de notas entre bimestres, asistencia por mes).
3. Probar *gradient boosting* y calibración de probabilidades, manteniendo la misma partición de prueba.
