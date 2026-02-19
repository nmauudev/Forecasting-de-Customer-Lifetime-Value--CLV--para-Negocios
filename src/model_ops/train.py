"""
train.py – Model Training & MLflow Tracking
============================================
Responsabilidad: ajustar los modelos probabilísticos de CLV y registrar
todo el experimento (hiperparámetros, métricas, artefactos) en MLflow.

Nota sobre serialización
------------------------
los objetos BetaGeoFitter y GammaGammaFitter contienen lambdas internas
creadas en el scope local de .fit() que pickle estándar no puede serializar.
Usamos `dill` (superconjunto de pickle) que maneja lambdas y closures.
Extension .joblib mantenida para respetar la interfaz solicitada.

Arquitectura de modelos
-----------------------
  BG/NBD  (BetaGeoFitter)      → predice FRECUENCIA futura de compras
  Gamma-Gamma (GammaGammaFitter)→ predice VALOR MONETARIO esperado por compra

Flujo
-----
  1. Cargar rfm_cal_holdout.parquet
  2. Separar repeat buyers (subconjunto que requiere Gamma-Gamma)
  3. with mlflow.start_run():
       a. Ajustar BG/NBD sobre datos de calibración
       b. Ajustar Gamma-Gamma sobre repeat buyers
       c. Predecir frecuencia y valor monetario en el periodo holdout
       d. Calcular RMSE y MAE a nivel cliente
       e. Log de hiperparámetros + métricas + modelos + params JSON
  4. Serializar ambos modelos con joblib

Entrada
-------
  data/processed/rfm_cal_holdout.parquet

Salidas
-------
  models/bg_nbd_model.joblib
  models/gamma_gamma_model.joblib
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import dill
import mlflow
import mlflow.sklearn        # para log_artifact compatibilidad
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")   # lifetimes emite deprecation warnings de scipy

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
INPUT_FILE  = ROOT / "data" / "processed" / "rfm_cal_holdout.parquet"
MODELS_DIR  = ROOT / "models"
BGF_PATH    = MODELS_DIR / "bg_nbd_model.joblib"
GGF_PATH    = MODELS_DIR / "gamma_gamma_model.joblib"

# ---------------------------------------------------------------------------
# Hiperparámetros configurables
# ---------------------------------------------------------------------------
BGF_PENALIZER: float = 0.001    # regularización L2 para BG/NBD
GGF_PENALIZER: float = 0.001    # regularización L2 para Gamma-Gamma
HOLDOUT_WEEKS: float = 26.0     # duración del periodo holdout (semanas)

EXPERIMENT_NAME = "CLV_BG_NBD_GammaGamma"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _log_json(obj: dict, filename: str, run_dir: Path) -> None:
    """Guarda un dict como JSON y lo registra como artefacto MLflow."""
    path = run_dir / filename
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    mlflow.log_artifact(str(path))
    log.info("Artefacto guardado: %s", path.name)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    bgf_penalizer: float = BGF_PENALIZER,
    ggf_penalizer: float = GGF_PENALIZER,
    holdout_weeks: float = HOLDOUT_WEEKS,
) -> tuple[BetaGeoFitter, GammaGammaFitter, dict]:
    """
    Entrena BG/NBD + Gamma-Gamma, valida contra holdout y registra en MLflow.

    Returns
    -------
    bgf       : modelo BG/NBD ajustado
    ggf       : modelo Gamma-Gamma ajustado
    metrics   : dict con las métricas de validación
    """

    # ------------------------------------------------------------------
    # 0. Preparación de directorios y MLflow
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "mlruns_tmp"
    tmp_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"file:///{(ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # 1. CARGA DE DATOS
    # ------------------------------------------------------------------
    log.info("=== FASE 1: Carga de datos ===")
    rfm = pd.read_parquet(INPUT_FILE)
    log.info("RFM cargado: shape=%s", rfm.shape)

    # Subconjunto de repeat buyers para Gamma-Gamma
    # (el modelo requiere frequency > 0 y monetary_value > 0)
    repeat_mask = (rfm["frequency_cal"] > 0) & (rfm["monetary_value_cal"] > 0)
    rfm_repeat  = rfm[repeat_mask].copy()
    log.info(
        "Repeat buyers con monetary_value > 0: %d  (%.1f%%)",
        len(rfm_repeat), 100 * len(rfm_repeat) / len(rfm),
    )

    # ------------------------------------------------------------------
    # 2. ENTRENAMIENTO DENTRO DEL RUN DE MLFLOW
    # ------------------------------------------------------------------
    log.info("=== FASE 2: Inicio del run MLflow — experimento: %s ===",
             EXPERIMENT_NAME)

    with mlflow.start_run(run_name="bg_nbd_gamma_gamma") as run:
        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

        # ── 2a. Log de hiperparámetros ─────────────────────────────────
        mlflow.log_params({
            "bgf_penalizer_coef": bgf_penalizer,
            "ggf_penalizer_coef": ggf_penalizer,
            "holdout_weeks":      holdout_weeks,
            "n_customers_total":  len(rfm),
            "n_repeat_buyers":    len(rfm_repeat),
        })

        # ── 2b. Ajuste BG/NBD ─────────────────────────────────────────
        log.info("=== FASE 2b: Ajustando BG/NBD (BetaGeoFitter) ===")
        bgf = BetaGeoFitter(penalizer_coef=bgf_penalizer)
        bgf.fit(
            frequency   = rfm["frequency_cal"],
            recency     = rfm["recency_cal"],
            T           = rfm["T_cal"],
        )
        log.info("BG/NBD ajustado. Params: %s", bgf.params_)
        mlflow.log_params({
            "bgf_r":     bgf.params_["r"],
            "bgf_alpha": bgf.params_["alpha"],
            "bgf_a":     bgf.params_["a"],
            "bgf_b":     bgf.params_["b"],
        })

        # ── 2c. Ajuste Gamma-Gamma ─────────────────────────────────────
        log.info("=== FASE 2c: Ajustando Gamma-Gamma (GammaGammaFitter) ===")
        ggf = GammaGammaFitter(penalizer_coef=ggf_penalizer)
        ggf.fit(
            frequency      = rfm_repeat["frequency_cal"],
            monetary_value = rfm_repeat["monetary_value_cal"],
        )
        log.info("Gamma-Gamma ajustado. Params: %s", ggf.params_)
        mlflow.log_params({
            "ggf_p": ggf.params_["p"],
            "ggf_q": ggf.params_["q"],
            "ggf_v": ggf.params_["v"],
        })

        # ── 2d. Predicciones en holdout ────────────────────────────────
        log.info("=== FASE 2d: Predicciones sobre periodo holdout ===")

        # Frecuencia predicha en el holdout
        rfm["predicted_purchases_holdout"] = bgf.predict(
            t           = holdout_weeks,
            frequency   = rfm["frequency_cal"],
            recency     = rfm["recency_cal"],
            T           = rfm["T_cal"],
        )

        # Valor monetario esperado (solo repeat buyers; imputamos 0 para one-timers)
        rfm["predicted_monetary"] = 0.0
        rfm.loc[repeat_mask, "predicted_monetary"] = ggf.conditional_expected_average_profit(
            frequency      = rfm_repeat["frequency_cal"],
            monetary_value = rfm_repeat["monetary_value_cal"],
        ).values

        # CLV esperado en el holdout = frecuencia × valor monetario
        rfm["predicted_clv_holdout"] = (
            rfm["predicted_purchases_holdout"] * rfm["predicted_monetary"]
        )

        # CLV real en holdout = frecuencia real × gasto real promedio
        rfm["actual_clv_holdout"] = (
            rfm["frequency_holdout"] * rfm["monetary_value_holdout"]
        )

        # ── 2e. Métricas de validación ─────────────────────────────────
        log.info("=== FASE 2e: Calculando metricas de validacion ===")

        # --- FRECUENCIA ------------------------------------------------
        freq_rmse = _rmse(rfm["frequency_holdout"], rfm["predicted_purchases_holdout"])
        freq_mae  = _mae(rfm["frequency_holdout"],  rfm["predicted_purchases_holdout"])

        # --- CLV -------------------------------------------------------
        clv_rmse = _rmse(rfm["actual_clv_holdout"], rfm["predicted_clv_holdout"])
        clv_mae  = _mae(rfm["actual_clv_holdout"],  rfm["predicted_clv_holdout"])

        # --- Correlación predicción vs real (repeat buyers) -----------
        mask_active_holdout = rfm["frequency_holdout"] > 0
        corr_freq = (
            rfm.loc[mask_active_holdout, "frequency_holdout"]
            .corr(rfm.loc[mask_active_holdout, "predicted_purchases_holdout"])
        )

        metrics = {
            "frequency_rmse":       freq_rmse,
            "frequency_mae":        freq_mae,
            "clv_rmse":             clv_rmse,
            "clv_mae":              clv_mae,
            "corr_freq_active":     corr_freq,
            "n_active_in_holdout":  int(mask_active_holdout.sum()),
        }

        mlflow.log_metrics(metrics)

        log.info("Metricas de validacion:")
        for k, v in metrics.items():
            log.info("  %-30s = %.6f", k, v)

        # ── 2f. Artefactos JSON con params de los modelos ──────────────
        _log_json(
            {"model": "BetaGeoFitter",
             "penalizer_coef": bgf_penalizer,
             "params": {k: float(v) for k, v in bgf.params_.items()}},
            "bgf_params.json",
            MODELS_DIR,
        )
        _log_json(
            {"model": "GammaGammaFitter",
             "penalizer_coef": ggf_penalizer,
             "params": {k: float(v) for k, v in ggf.params_.items()}},
            "ggf_params.json",
            MODELS_DIR,
        )

        # ── 2g. Serialización con dill ───────────────────────────────
        # dill puede serializar lambdas y closures locales de lifetimes.
        log.info("=== FASE 2g: Serializando modelos con dill ===")
        with open(BGF_PATH, "wb") as f:
            dill.dump(bgf, f)
        with open(GGF_PATH, "wb") as f:
            dill.dump(ggf, f)
        log.info("Modelo BG/NBD guardado en:      %s", BGF_PATH)
        log.info("Modelo Gamma-Gamma guardado en: %s", GGF_PATH)

        # Registrar los .joblib como artefactos MLflow también
        mlflow.log_artifact(str(BGF_PATH),  artifact_path="models")
        mlflow.log_artifact(str(GGF_PATH),  artifact_path="models")

        log.info("Run MLflow completado. run_id=%s", run_id)

    return bgf, ggf, metrics


# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------

def print_summary(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    metrics: dict,
) -> None:
    print("\n" + "=" * 65)
    print("  RESUMEN DEL ENTRENAMIENTO")
    print("=" * 65)

    print("\n  [BG/NBD — BetaGeoFitter]")
    for k, v in bgf.params_.items():
        print(f"    {k:<10} = {v:.8f}")

    print("\n  [Gamma-Gamma — GammaGammaFitter]")
    for k, v in ggf.params_.items():
        print(f"    {k:<10} = {v:.8f}")

    print("\n  [Metricas de validacion en holdout (26 semanas)]")
    print(f"    Frequency  RMSE  = {metrics['frequency_rmse']:.6f}")
    print(f"    Frequency  MAE   = {metrics['frequency_mae']:.6f}")
    print(f"    CLV        RMSE  = {metrics['clv_rmse']:.4f}")
    print(f"    CLV        MAE   = {metrics['clv_mae']:.4f}")
    print(f"    Corr freq (activos) = {metrics['corr_freq_active']:.4f}")
    print(f"    Clientes activos en holdout = {metrics['n_active_in_holdout']:,}")

    print(f"\n  Modelos guardados en:  models/")
    print(f"    bg_nbd_model.joblib")
    print(f"    gamma_gamma_model.joblib")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bgf, ggf, metrics = train()
    print_summary(bgf, ggf, metrics)
