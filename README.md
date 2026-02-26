#  CLV Forecasting — Customer Lifetime Value para Negocios

Sistema de predicción de **Customer Lifetime Value (CLV)** basado en modelos probabilísticos BG/NBD y Gamma-Gamma, con API REST, interfaz web Streamlit y monitoreo de deriva de datos con Evidently AI.

---

##  Arquitectura del Sistema

```
┌─────────────────┐   HTTP requests    ┌────────────────────────┐
│  Streamlit App  │ ─────────────────► │  FastAPI Backend       │
│  :8501          │ ◄───────────────── │  :8000                 │
└─────────────────┘   JSON responses   │  /predict-clv          │
                                       │  /customer/{id}        │
                                       │  /monitoring/stats     │
                                       └──────────┬─────────────┘
                                                  │ logs c/ request
                                       ┌──────────▼─────────────┐
                                       │  data/production_logs/ │
                                       │  requests.db (SQLite)  │
                                       │  requests.csv (backup) │
                                       └──────────┬─────────────┘
                                                  │
                                       ┌──────────▼─────────────┐
                                       │  Monitor Evidently AI  │
                                       │  src/monitoring/       │
                                       │  monitor.py            │
                                       └──────────┬─────────────┘
                                                  │
                                       ┌──────────▼─────────────┐
                                       │  reports/drift/        │
                                       │  drift_report_<ts>.html│
                                       └────────────────────────┘
```

##  Estructura del Proyecto

```
 Forecasting de Customer Lifetime Value (CLV) para Negocios/
│
├──  .github/workflows/
│   └── ci.yml                    ← Pipeline CI (GitHub Actions)
│
├──  data/
│   ├── raw/                      ← CSVs originales del dataset Olist
│   ├── processed/                ← Parquets generados por el pipeline
│   │   ├── clean_transactions.parquet
│   │   ├── rfm_cal_holdout.parquet
│   │   └── rfm_clv.parquet       ← tabla RFM completa (93k clientes)
│   └── production_logs/          ← Logs de requests de la API (auto)
│       ├── requests.db           ← SQLite (fuente principal)
│       └── requests.csv          ← CSV backup legible
│
├──  models/
│   ├── bg_nbd_model.joblib       ← Modelo BG/NBD entrenado
│   └── gamma_gamma_model.joblib  ← Modelo Gamma-Gamma entrenado
│
├──  notebooks/
│   └── rfm_y_modelado_clv.py     ← Script de modelado exploratorio
│
├──  reports/
│   └── drift/                    ← Reportes de deriva generados
│       ├── drift_report_<ts>.html
│       └── drift_summary_<ts>.json
│
├──  src/
│   ├── api/
│   │   └── main.py               ← FastAPI app (endpoints + logging)
│   ├── app/
│   │   └── app.py                ← Streamlit frontend
│   ├── data_pipeline/
│   │   └── etl.py                ← Pipeline ETL (Olist → RFM)
│   ├── feature_engineering/
│   │   └── build_features.py     ← Construcción de features RFM
│   ├── model_ops/
│   │   └── train.py              ← Entrenamiento BG/NBD + Gamma-Gamma
│   └── monitoring/
│       └── monitor.py            ← Monitor de data drift (Evidently)
│
├──  tests/
│   ├── test_api.py               ← Tests del backend FastAPI
│   └── test_data.py              ← Tests del pipeline ETL
│
├── Dockerfile                    ← Imagen Docker del backend
├── Dockerfile.streamlit          ← Imagen Docker del frontend
├── docker-compose.yml            ← Orquestación de ambos servicios
├── requirements.txt              ← Deps completas (desarrollo)
├── requirements.api.txt          ← Deps del backend (Docker)
└── requirements.streamlit.txt    ← Deps del frontend (Docker)
```

---

##  Inicio Rápido

### Prerrequisitos

- Python 3.10+
- Git
- (Opcional) Docker y Docker Compose

### Instalación Local

```bash
# Clonar el repositorio
git clone <repository-url>
cd "Forecasting de Customer Lifetime Value (CLV) para Negocios"

# Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Con Docker

```bash
# Construir y levantar ambos servicios (API + Streamlit)
docker-compose up --build

# Solo el backend
docker-compose up clv_api

# Solo el frontend
docker-compose up clv_streamlit
```

---

##  Uso del Sistema

### 1. Correr el Pipeline de Datos

```bash
python -m src.data_pipeline.etl
```
Genera `data/processed/clean_transactions.parquet` a partir de los CSVs en `data/raw/`.

### 2. Entrenar los Modelos

```bash
python -m src.model_ops.train
```
Entrena los modelos BG/NBD y Gamma-Gamma y los guarda en `models/`.

### 3. Levantar la API Backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints disponibles:
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/predict-clv` | Predice CLV a 12 meses dado RFM |
| `GET`  | `/customer/{id}` | Datos RFM de un cliente por ID |
| `GET`  | `/customers/sample` | Muestra aleatoria de clientes |
| `GET`  | `/monitoring/stats` | Estadísticas de logs de producción |
| `GET`  | `/health` | Healthcheck de la API |
| `GET`  | `/docs` | Documentación interactiva (Swagger) |

### 4. Levantar el Frontend Streamlit

```bash
streamlit run src/app/app.py
```
Abre en `http://localhost:8501`

### 5. Correr el Monitor de Deriva

```bash
# Con datos reales de producción (requiere ≥30 requests a la API)
python -m src.monitoring.monitor

# Modo demo con datos sintéticos (para probar el reporte)
python -m src.monitoring.monitor --demo

# Especificar ruta de salida
python -m src.monitoring.monitor --demo --output reports/mi_reporte.html
```

Genera `reports/drift/drift_report_<timestamp>.html` con análisis visual interactivo de deriva.

---

##  Tests

```bash
# Correr todos los tests
pytest tests/ -v

# Con cobertura de código
pytest tests/ --cov=src --cov-report=term-missing

# Solo tests del backend
pytest tests/test_api.py -v

# Solo tests del pipeline de datos
pytest tests/test_data.py -v
```

---

##  CI Pipeline (GitHub Actions)

El pipeline `.github/workflows/ci.yml` se ejecuta automáticamente en cada `push` y `pull_request` a `main`/`develop`:

1. **Setup** — Python 3.11 + caché de dependencias
2. **Instala deps** — `requirements.api.txt` + `requirements.streamlit.txt`
3. **Valida compilación** — verifica que `main.py` y `app.py` importan correctamente
4. **Tests API** — `pytest tests/test_api.py`
5. **Tests ETL** — `pytest tests/test_data.py`
6. **Reporte** — sube resultados de test como artefactos

---

##  Metodología CLV

### Modelos Implementados

| Modelo | Librería | Descripción |
|--------|----------|-------------|
| **BG/NBD** | `lifetimes` | Predice número de transacciones futuras |
| **Gamma-Gamma** | `lifetimes` | Predice valor monetario esperado |
| **CLV = BG/NBD × Gamma-Gamma** | — | CLV a 12 meses en pesos |

### Pipeline de Features (RFM)

```
Olist CSVs → ETL → clean_transactions.parquet → build_features.py → rfm_clv.parquet
```

- **Recency (R)**: días desde la última compra
- **Frequency (F)**: número de compras repetidas
- **Monetary (M)**: ticket promedio en pesos
- **T**: antigüedad del cliente en días

### Monitoreo de Deriva

Comparación continua entre los datos de entrenamiento y los requests reales que recibe la API:

```
rfm_clv.parquet (referencia)  ─┐
                                ├── Evidently DataDriftPreset → drift_report.html
requests.db (producción)      ─┘
```

Test estadístico: **Kolmogorov-Smirnov** por columna (p < 0.05 → alerta de deriva).

---

##  Stack Tecnológico

| Área | Herramientas |
|------|-------------|
| **ML/Estadística** | `lifetimes`, `scikit-learn`, `scipy`, `statsmodels` |
| **Datos** | `pandas`, `numpy`, `pyarrow` |
| **Backend API** | `fastapi`, `uvicorn`, `pydantic` |
| **Frontend** | `streamlit`, `plotly` |
| **Monitoreo** | `evidently` (v0.7+) |
| **MLOps** | `mlflow`, `dill` |
| **Testing** | `pytest`, `starlette.testclient`, `httpx` |
| **CI/CD** | GitHub Actions |
| **Contenedores** | Docker, Docker Compose |
| **Logging prod.** | SQLite + CSV |

---

## Dataset

Este proyecto usa el dataset público **[Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** de Kaggle.

Archivos requeridos en `data/raw/`:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`

---

## Referencias

- Fader, P.S. & Hardie, B.G.S. (2005): *"A Note on Deriving the Pareto/NBD Model"*
- Fader, P.S., Hardie, B.G.S. & Lee, K.L. (2005): *"RFM and CLV"* — Journal of Marketing Research
- [Lifetimes Documentation](https://lifetimes.readthedocs.io/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
