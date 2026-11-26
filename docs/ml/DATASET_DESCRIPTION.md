# Descripción del dataset

## Origen
**Sintético.** No existe un dataset autorizado de estudiantes reales (Ley N° 29733 de Protección de Datos Personales), así que el modelo se entrena con datos ficticios generados por `ml-service/pipeline/dataset.py` (`generar(n=2000, seed=42)`). El generador es el mismo que el proyecto usaba desde su primera versión (`train_dummy.py`); **no se ajustó para alcanzar las metas de RNF‑02**.

## Tamaño y balance
| Elemento | Valor |
|---|---|
| Filas | 2 000 |
| Entrenamiento / prueba | 1 600 / 400 (estratificado, semilla 42) |
| Clase positiva (`en_riesgo = 1`) | 28,3 % |
| Balanceo | SMOTE solo en los *folds* de entrenamiento |

## Variables
| Variable | Tipo | Distribución de generación | Origen en el sistema |
|---|---|---|---|
| `promedio` | continua 0–20 | Normal(13; 2,5) recortada | Promedio de notas del bimestre |
| `inasistencias` | entera | Poisson(3) | Registros de asistencia |
| `conducta` | ordinal 1–3 | 60 % / 30 % / 10 % | Conducta registrada (1 = buena) |
| `horas_estudio` | continua | Gamma(2; 2) recortada a 0–20 | Encuesta familiar |
| `nivel_edu_padres` | ordinal 1–3 | 20 % / 50 % / 30 % | Ficha del apoderado |
| `acceso_tecnologia` | binaria | 25 % sin / 75 % con | Ficha del apoderado |
| `tiempo_disponible_padres` | ordinal 1–3 | uniforme | Encuesta familiar |
| `comunicacion_familiar` | ordinal 1–3 | 25 % / 45 % / 30 % | Encuesta familiar |
| `cursos_desaprobados` | entera | Poisson dependiente del promedio | Notas < 11 |
| `tareas_entregadas_pct` | continua 0–1 | Beta(5; 2) | Registro de tareas |

## Variable objetivo
`en_riesgo` (0/1): el estudiante desaprueba el año o abandona. Se simula con un puntaje latente lineal de las variables (peor promedio, más inasistencias, peor conducta y más cursos desaprobados aumentan el riesgo; más horas de estudio, comunicación familiar, nivel educativo de los padres y tareas entregadas lo reducen) y un sorteo de Bernoulli con probabilidad sigmoide. El sorteo introduce ruido realista: dos perfiles iguales pueden tener desenlaces distintos.

## Limitaciones
- Las relaciones son las que el generador impone; el modelo no aprende patrones de una población real.
- Las distribuciones no se calibraron con estadísticas del MINEDU.
- Los resultados (F1, AUC) solo describen el desempeño sobre este proceso sintético.
- En la demo, el backend calcula las variables a partir de datos del seed, también sintéticos.
