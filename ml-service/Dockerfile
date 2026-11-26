FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY pipeline ./pipeline
COPY model ./model
# Entrena el pipeline (VIF + SMOTE + Random Forest) con el dataset sintético reproducible.
# --forzar: el modelo actual NO cumple F1 >= 0.75 en prueba (ver reports/metricas.json); se publica solo
# para la demostración académica y GET /modelo expone sus métricas reales.
RUN [ -f model/model.joblib ] || python -m pipeline.entrenar --forzar
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
