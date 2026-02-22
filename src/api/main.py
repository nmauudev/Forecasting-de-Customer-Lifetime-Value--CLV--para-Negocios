"""
CLV Forecasting API — Motor de predicción
==========================================
Levanta en el puerto 8000.
  POST /predict-clv   → recibe RFM y devuelve CLV a 12 meses
  GET  /customer/{id} → devuelve los valores RFM de un cliente conocido
  GET  /health        → healthcheck
"""

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

BG_NBD_PATH       = MODEL_DIR / "bg_nbd_model.joblib"
GAMMA_GAMMA_PATH  = MODEL_DIR / "gamma_gamma_model.joblib"
RFM_CLV_PATH      = DATA_DIR  / "rfm_clv.parquet"

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

        return CLVOutput(
            clv_predicted=round(clv_predicted, 4),
            expected_purchases=round(expected_purchases, 6),
            expected_avg_revenue=round(expected_avg_revenue, 4),
            prob_alive=round(prob_alive, 6),
            horizon_months=data.months,
        )

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
