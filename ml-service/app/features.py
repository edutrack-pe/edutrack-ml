"""Variables predictoras compartidas entre entrenamiento y predicción.

El orden importa: el modelo se entrena con este orden de columnas.
"""

FEATURES: list[str] = [
    "promedio",                  # promedio ponderado del bimestre (0-20)
    "inasistencias",             # número de faltas en el bimestre
    "conducta",                  # 1 = buena, 2 = regular, 3 = mala
    "horas_estudio",             # horas de estudio semanales en casa
    "nivel_edu_padres",          # 1 = primaria, 2 = secundaria, 3 = superior
    "acceso_tecnologia",         # 0 = no, 1 = sí
    "tiempo_disponible_padres",  # 1 = bajo, 2 = medio, 3 = alto
    "comunicacion_familiar",     # 1 = baja, 2 = media, 3 = alta
    "cursos_desaprobados",       # cursos con nota < 11
    "tareas_entregadas_pct",     # 0.0 - 1.0
]
