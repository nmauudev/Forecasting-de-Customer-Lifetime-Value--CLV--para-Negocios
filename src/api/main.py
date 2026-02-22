"""
CLV Forecasting API — Motor de predicción
==========================================
Levanta en el puerto 8000.
  POST /predict-clv        → recibe RFM y devuelve CLV a 12 meses
  GET  /customer/{id}      → devuelve los valores RFM de un cliente conocido
  GET  /health             → healthcheck
  GET  /monitoring/stats   → estadísticas de logs de producción

Logging de producción
---------------------
Cada request a /predict-clv queda registrado en:
  • data/production_logs/requests.db   (SQLite — fuente principal)
  • data/production_logs/requests.csv  (CSV plano — backup legible)
Estos archivos los consume src/monitoring/monitor.py para detectar deriva.
"""

import csv
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuración de logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Rutas (relativas a la raíz del proyecto)
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]   # raíz del proyecto
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data" / "processed"
LOGS_DIR  = BASE_DIR / "data" / "production_logs"

BG_NBD_PATH       = MODEL_DIR / "bg_nbd_model.joblib"
GAMMA_GAMMA_PATH  = MODEL_DIR / "gamma_gamma_model.joblib"
RFM_CLV_PATH      = DATA_DIR  / "rfm_clv.parquet"

# Rutas de logs de producción (para monitoreo con Evidently)
PROD_DB_PATH  = LOGS_DIR / "requests.db"
PROD_CSV_PATH = LOGS_DIR / "requests.csv"

# Lock para escrituras concurrentes seguras
_db_lock = threading.Lock()

# ──────────────────────────────────────────────
# Carga de modelos (una sola vez al arrancar)
# ──────────────────────────────────────────────
logger.info("Cargando modelos…")
try:
    bg_nbd_model    = joblib.load(BG_NBD_PATH)
    gamma_gamma_model = joblib.load(GAMMA_GAMMA_PATH)
    logger.info("✅ Modelos cargados correctamente.")
except Exception as exc:
    logger.error("❌ Error al cargar los modelos: %s", exc)
    raise RuntimeError(f"No se pudieron cargar los modelos: {exc}") from exc

# ──────────────────────────────────────────────
# Carga de la tabla RFM pre-computada
# ──────────────────────────────────────────────
logger.info("Cargando tabla RFM…")
try:
    rfm_df = pd.read_parquet(RFM_CLV_PATH)
    logger.info("✅ Tabla RFM cargada: %d clientes.", len(rfm_df))
except Exception as exc:
    logger.warning("⚠️  No se pudo cargar la tabla RFM: %s. El endpoint /customer no estará disponible.", exc)
    rfm_df = None

# ──────────────────────────────────────────────
# Logging de producción — SQLite + CSV
# ──────────────────────────────────────────────

def _init_production_logs() -> None:
    """
    Crea el directorio y la tabla SQLite la primera vez que arranca la API.
    También inicializa el CSV con cabecera si no existe.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # SQLite
    with sqlite3.connect(str(PROD_DB_PATH)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                frequency       REAL    NOT NULL,
                recency         REAL    NOT NULL,
                T               REAL    NOT NULL,
                monetary_value  REAL    NOT NULL,
                months          INTEGER NOT NULL,
                clv_predicted   REAL,
                expected_purchases REAL,
                expected_avg_revenue REAL,
                prob_alive      REAL
            )
        """)
        con.commit()

    # CSV (cabecera solo si el archivo es nuevo)
    if not PROD_CSV_PATH.exists() or PROD_CSV_PATH.stat().st_size == 0:
        with open(PROD_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frequency", "recency", "T", "monetary_value",
                "months", "clv_predicted", "expected_purchases",
                "expected_avg_revenue", "prob_alive",
            ])

    logger.info("Logging de produccion inicializado en: %s", LOGS_DIR)


def _log_request(
    data: "RFMInput",
    result: "CLVOutput | None" = None,
) -> None:
    """
    Registra un request en SQLite y CSV de forma thread-safe.
    Falla silenciosamente para no interrumpir la respuesta al cliente.
    """
    ts = datetime.now(tz=timezone.utc).isoformat()

    clv          = result.clv_predicted          if result else None
    exp_purch    = result.expected_purchases      if result else None
    exp_rev      = result.expected_avg_revenue    if result else None
    prob_alive   = result.prob_alive              if result else None

    row_values = (
        ts,
        data.frequency,
        data.recency,
        data.T,
        data.monetary_value,
        data.months,
        clv,
        exp_purch,
        exp_rev,
        prob_alive,
    )

    try:
        with _db_lock:
            # ── SQLite ────────────────────────────────────────────────────
            with sqlite3.connect(str(PROD_DB_PATH)) as con:
                con.execute("""
                    INSERT INTO prediction_logs (
                        timestamp, frequency, recency, T, monetary_value,
                        months, clv_predicted, expected_purchases,
                        expected_avg_revenue, prob_alive
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row_values)
                con.commit()

            # ── CSV (backup) ──────────────────────────────────────────────
            with open(PROD_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row_values)

    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo registrar el request en los logs: %s", exc)


# Inicializar sistema de logs al arrancar
try:
    _init_production_logs()
except Exception as _log_init_exc:
    logger.warning("No se pudo inicializar el sistema de logs: %s", _log_init_exc)

# ──────────────────────────────────────────────
# Aplicación FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="CLV Forecasting API",
    description="Motor de predicción de Customer Lifetime Value a 12 meses usando BG/NBD + Gamma-Gamma.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Esquemas de datos
# ──────────────────────────────────────────────

class RFMInput(BaseModel):
    """
    Parámetros RFM de un cliente.

    - frequency:      número de compras repetidas (compras_totales - 1).
    - recency:        días entre la primera y la última compra.
    - T:              días desde la primera compra hasta el corte de observación.
    - monetary_value: ticket promedio (valor monetario promedio por compra).
    """
    frequency:      float = Field(..., ge=0,   description="Número de compras repetidas (≥ 0).")
    recency:        float = Field(..., ge=0,   description="Días desde la primera hasta la última compra (≥ 0).")
    T:              float = Field(..., gt=0,   description="Días de antigüedad del cliente (> 0).")
    monetary_value: float = Field(..., ge=0,   description="Ticket promedio en USD (≥ 0).")
    months:         int   = Field(12,  gt=0,   description="Horizonte de predicción en meses (por defecto 12).")


class CLVOutput(BaseModel):
    clv_predicted:        float
    expected_purchases:   float
    expected_avg_revenue: float
    prob_alive:           float
    horizon_months:       int


class CustomerRFM(BaseModel):
    customer_unique_id:    str
    frequency:             float
    recency:               float
    T:                     float
    monetary_value:        float
    clv_12m_precomputed:   float
    prob_activo:           float


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", tags=["Utilidades"])
def health_check():
    """Verifica que la API y los modelos están operativos."""
    return {"status": "ok", "models_loaded": True}


@app.post("/predict-clv", response_model=CLVOutput, tags=["Predicción"])
def predict_clv(data: RFMInput):
    """
    Recibe los valores RFM de un cliente y devuelve el CLV predicho
    para el horizonte de `months` meses (por defecto 12).
    """
    try:
        # ── 1. Construir el DataFrame de entrada en el formato que esperan los modelos
        rfm_row = pd.DataFrame([{
            "frequency":      data.frequency,
            "recency":        data.recency,
            "T":              data.T,
            "monetary_value": data.monetary_value,
        }])

        def _to_float(val) -> float:
            """Convierte Series / ndarray / scalar a float de Python."""
            if hasattr(val, "iloc"):
                return float(val.iloc[0])
            if hasattr(val, "item"):
                return float(val.item())
            return float(val)

        # ── 2. Predicciones con BG/NBD
        expected_purchases = _to_float(
            bg_nbd_model.predict(
                t=data.months * 30,        # convertir meses → días aproximados
                frequency=rfm_row["frequency"],
                recency=rfm_row["recency"],
                T=rfm_row["T"],
            )
        )

        prob_alive = _to_float(
            bg_nbd_model.conditional_probability_alive(
                frequency=rfm_row["frequency"],
                recency=rfm_row["recency"],
                T=rfm_row["T"],
            )
        )

        # ── 3. Valor monetario esperado con Gamma-Gamma
        #      (solo tiene sentido si el cliente ha comprado al menos una vez)
        if data.frequency > 0 and data.monetary_value > 0:
            expected_avg_revenue = _to_float(
                gamma_gamma_model.conditional_expected_average_profit(
                    frequency=rfm_row["frequency"],
                    monetary_value=rfm_row["monetary_value"],
                )
            )
        else:
            expected_avg_revenue = float(data.monetary_value)

        # ── 4. CLV = compras esperadas × revenue esperado por compra
        #          (ajuste de descuento ya incorporado en el modelo GG)
        clv_predicted = expected_purchases * expected_avg_revenue

        logger.info(
            "Predicción — freq=%.1f rec=%.1f T=%.1f mon=%.2f → CLV=%.4f",
            data.frequency, data.recency, data.T, data.monetary_value, clv_predicted,
        )

        output = CLVOutput(
            clv_predicted=round(clv_predicted, 4),
            expected_purchases=round(expected_purchases, 6),
            expected_avg_revenue=round(expected_avg_revenue, 4),
            prob_alive=round(prob_alive, 6),
            horizon_months=data.months,
        )

        # ── Guardar request en logs de producción (para monitoreo) ────────
        _log_request(data, output)

        return output

    except Exception as exc:
        logger.exception("Error durante la predicción.")
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {exc}") from exc


@app.get("/customer/{customer_unique_id}", response_model=CustomerRFM, tags=["Clientes"])
def get_customer(customer_unique_id: str):
    """
    Devuelve los valores RFM pre-computados de un cliente de Olist por su ID.
    Útil para pre-llenar los sliders en el frontend.
    """
    if rfm_df is None:
        raise HTTPException(status_code=503, detail="La tabla RFM no está disponible.")

    if customer_unique_id not in rfm_df.index:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente '{customer_unique_id}' no encontrado en la base de datos.",
        )

    row = rfm_df.loc[customer_unique_id]
    return CustomerRFM(
        customer_unique_id=customer_unique_id,
        frequency=float(row["frequency"]),
        recency=float(row["recency"]),
        T=float(row["T"]),
        monetary_value=float(row["monetary_value"]),
        clv_12m_precomputed=float(row["clv_12m"]),
        prob_activo=float(row["prob_activo"]),
    )


@app.get("/customers/sample", tags=["Clientes"])
def get_sample_customers(n: int = 10):
    """
    Devuelve una muestra aleatoria de IDs de clientes disponibles.
    Útil para testear el frontend sin conocer IDs de memoria.
    """
    if rfm_df is None:
        raise HTTPException(status_code=503, detail="La tabla RFM no está disponible.")

    sample = rfm_df.sample(min(n, len(rfm_df))).reset_index()[["customer_unique_id", "clv_12m", "frequency"]]
    return sample.to_dict(orient="records")


@app.get("/monitoring/stats", tags=["Monitoreo"])
def monitoring_stats():
    """
    Devuelve estadísticas básicas de los logs de producción:
    total de requests, rango de fechas, y estadísticas RFM.
    Útil para saber si hay suficientes datos para correr el monitor de deriva.
    """
    if not PROD_DB_PATH.exists():
        return {
            "status": "sin_datos",
            "message": "Todavía no hay logs de producción. Hacé algunas predicciones primero.",
            "total_requests": 0,
        }

    try:
        with sqlite3.connect(str(PROD_DB_PATH)) as con:
            total = con.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
            if total == 0:
                return {"status": "sin_datos", "total_requests": 0}

            row = con.execute("""
                SELECT
                    MIN(timestamp)          AS first_request,
                    MAX(timestamp)          AS last_request,
                    AVG(frequency)          AS avg_frequency,
                    AVG(recency)            AS avg_recency,
                    AVG(T)                  AS avg_T,
                    AVG(monetary_value)     AS avg_monetary_value,
                    AVG(clv_predicted)      AS avg_clv_predicted
                FROM prediction_logs
            """).fetchone()

        return {
            "status": "ok",
            "total_requests":       total,
            "first_request":        row[0],
            "last_request":         row[1],
            "avg_frequency":        round(row[2] or 0, 4),
            "avg_recency":          round(row[3] or 0, 4),
            "avg_T":                round(row[4] or 0, 4),
            "avg_monetary_value":   round(row[5] or 0, 4),
            "avg_clv_predicted":    round(row[6] or 0, 4),
            "db_path":              str(PROD_DB_PATH),
            "csv_path":             str(PROD_CSV_PATH),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error leyendo logs: {exc}") from exc
