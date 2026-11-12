# ml-research

Experimentación del modelo predictivo de EduTrack (OE1 y OE4).

| Notebook | Paso |
|---|---|
| `01_eda.ipynb` | Análisis exploratorio, balance de clases, correlaciones |
| `02_vif_smote.ipynb` | Multicolinealidad (VIF > 5 se descarta), split train/test y SMOTE solo en train |
| `03_entrenamiento_rf.ipynb` | GridSearch del Random Forest optimizando F1 y exportación a `ml-service/model/` |
| `04_evaluacion.ipynb` | Matriz de confusión, F1, curvas ROC y PR sobre el test |

## Metas (IND-06)

- **F1-Score ≥ 0.75**
- **AUC-ROC > 0.80**
- Mínimo **10 variables** académicas y sociofamiliares (IND-01)

## Datos

Coloca el dataset en `data/dataset.csv` con las columnas de [`ml-service/app/features.py`](../ml-service/app/features.py) más la columna objetivo `en_riesgo` (0/1).

> ⚠️ **Ley N° 29733:** la carpeta `data/` está ignorada en Git. Nunca subas datos reales de estudiantes; usa datos anonimizados.

## Entorno

```bash
cd ml-research
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
