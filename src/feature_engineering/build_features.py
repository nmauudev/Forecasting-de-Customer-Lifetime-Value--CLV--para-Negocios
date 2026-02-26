from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from lifetimes.utils import calibration_and_holdout_data

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
ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE  = ROOT / "data" / "processed" / "clean_transactions.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "rfm_cal_holdout.parquet"

# ---------------------------------------------------------------------------
# Parámetros del corte temporal
# ---------------------------------------------------------------------------
# Frecuencia de agregación temporal usada en lifetimes
# 'W' = semanas  →  unidad estándar para BG/NBD y Gamma-Gamma
FREQ = "W"

# Fin del periodo de calibración  (los últimos ~6 meses son holdout)
CUTOFF_DATE: str = "2018-02-28"

# Fin del periodo de observación completo (ultimo dato del dataset)
OBS_END_DATE: str = "2018-08-29"

# Columnas del dataset de transacciones que usaremos
COL_CUSTOMER   = "customer_unique_id"
COL_DATETIME   = "order_purchase_timestamp"
COL_MONETARY   = "revenue"          # price + freight_value


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_rfm_cal_holdout(
    cutoff_date: str = CUTOFF_DATE,
    obs_end_date: str = OBS_END_DATE,
    freq: str = FREQ,
) -> pd.DataFrame:
    """
    Construye la matriz RFM calibración/holdout y la persiste en parquet.

    Parameters
    ----------
    cutoff_date   : str  – fecha de corte 'YYYY-MM-DD'  (fin de calibración)
    obs_end_date  : str  – fecha de fin de observación 'YYYY-MM-DD' (fin holdout)
    freq          : str  – unidad temporal de lifetimes ('D', 'W', 'M')

    Returns
    -------
    pd.DataFrame  – tabla RFM lista para modelado
    """

    # ------------------------------------------------------------------
    # 1. CARGA
    # ------------------------------------------------------------------
    log.info("=== FASE 1: Carga de transacciones limpias ===")
    transactions = pd.read_parquet(INPUT_FILE)
    log.info("Transacciones cargadas: shape=%s", transactions.shape)
    log.info(
        "Rango de fechas: %s -> %s",
        transactions[COL_DATETIME].min().date(),
        transactions[COL_DATETIME].max().date(),
    )

    # ------------------------------------------------------------------
    # 2. VERIFICACION DEL CORTE TEMPORAL
    # ------------------------------------------------------------------
    log.info("=== FASE 2: Parametros del corte temporal ===")
    cutoff_ts  = pd.Timestamp(cutoff_date)
    obs_end_ts = pd.Timestamp(obs_end_date)

    cal_txns = transactions[transactions[COL_DATETIME] <= cutoff_ts]
    hld_txns = transactions[
        (transactions[COL_DATETIME] > cutoff_ts) &
        (transactions[COL_DATETIME] <= obs_end_ts)
    ]

    holdout_weeks = (obs_end_ts - cutoff_ts).days / 7

    log.info("Fecha de corte (fin calibracion): %s", cutoff_date)
    log.info("Fin de observacion   (holdout):   %s", obs_end_date)
    log.info("Duracion del holdout:             %.1f semanas (~%.0f meses)",
             holdout_weeks, holdout_weeks / 4.33)
    log.info("Transacciones en calibracion:     %d  (%.1f%%)",
             len(cal_txns), 100 * len(cal_txns) / len(transactions))
    log.info("Transacciones en holdout:         %d  (%.1f%%)",
             len(hld_txns), 100 * len(hld_txns) / len(transactions))

    # ------------------------------------------------------------------
    # 3. AGREGACION NIVEL ORDEN
    #    lifetimes espera UNA fila por transaccion (orden).
    #    Como nuestro parquet tiene una fila por ITEM, primero
    #    colapsamos al nivel orden sumando el revenue.
    # ------------------------------------------------------------------
    log.info("=== FASE 3: Agregacion a nivel de orden ===")
    order_level = (
        transactions
        .groupby([COL_CUSTOMER, "order_id", COL_DATETIME], as_index=False)
        [COL_MONETARY]
        .sum()
    )
    log.info("Ordenes unicas (nivel transaccion): %d", len(order_level))

    # ------------------------------------------------------------------
    # 4. CALIBRACION / HOLDOUT mediante lifetimes
    #    calibration_and_holdout_data devuelve por cliente:
    #      frequency_cal      – compras repetidas durante calibracion
    #      recency_cal        – semanas entre 1ra y ultima compra en cal.
    #      T_cal              – antiguedad al cutoff (semanas)
    #      monetary_value_cal – gasto medio por repeticion en cal.
    #      frequency_holdout  – compras en holdout
    #      duration_holdout   – semanas de holdout
    # ------------------------------------------------------------------
    log.info("=== FASE 4: Generando matriz RFM calibracion/holdout ===")
    rfm = calibration_and_holdout_data(
        transactions          = order_level,
        customer_id_col       = COL_CUSTOMER,
        datetime_col          = COL_DATETIME,
        calibration_period_end= cutoff_date,
        observation_period_end= obs_end_date,
        freq                  = freq,
        monetary_value_col    = COL_MONETARY,
    )

    log.info("Matriz RFM generada: shape=%s", rfm.shape)
    log.info("Columnas: %s", list(rfm.columns))

    # ------------------------------------------------------------------
    # 5. RESET INDEX  (customer_unique_id queda como columna)
    # ------------------------------------------------------------------
    rfm = rfm.reset_index()
    rfm.rename(columns={"index": COL_CUSTOMER}, inplace=True)

    # Asegurar nombre correcto de index si lifetimes lo llama distinto
    if COL_CUSTOMER not in rfm.columns:
        rfm.columns = [COL_CUSTOMER] + list(rfm.columns[1:])

    # ------------------------------------------------------------------
    # 6. FILTRO DE CALIDAD
    #    Descartamos clientes con T_cal == 0 (solo 1 punto en el tiempo,
    #    imposible calcular recency relativa) y monetary_value_cal <= 0.
    # ------------------------------------------------------------------
    log.info("=== FASE 5: Filtros de calidad ===")
    antes = len(rfm)

    # T_cal debe ser > 0 para que el modelo tenga sentido
    rfm = rfm[rfm["T_cal"] > 0]
    log.info("Clientes eliminados por T_cal == 0:              %d", antes - len(rfm))

    antes = len(rfm)
    # Gamma-Gamma requiere monetary_value_cal > 0 con compras repetidas
    # (frequency_cal > 0). Los one-time buyers se conservan para BG/NBD.
    rfm = rfm[rfm["monetary_value_cal"] >= 0]
    log.info("Clientes eliminados por monetary_value_cal < 0:  %d", antes - len(rfm))

    log.info("Clientes finales en la matriz RFM: %d", len(rfm))

    # ------------------------------------------------------------------
    # 7. COLUMNA AUXILIAR: flag de cliente repeat buyer en calibracion
    # ------------------------------------------------------------------
    rfm["is_repeat_buyer_cal"] = (rfm["frequency_cal"] > 0).astype(int)
    log.info(
        "Repeat buyers en calibracion: %d  (%.1f%%)",
        rfm["is_repeat_buyer_cal"].sum(),
        100 * rfm["is_repeat_buyer_cal"].mean(),
    )

    # ------------------------------------------------------------------
    # 8. GUARDADO
    # ------------------------------------------------------------------
    log.info("=== FASE 6: Guardando rfm_cal_holdout.parquet ===")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    log.info("Archivo guardado en: %s", OUTPUT_FILE)

    return rfm


# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------

def print_summary(rfm: pd.DataFrame) -> None:
    """Imprime estadisticas descriptivas del dataset RFM generado."""
    cols_num = ["frequency_cal", "recency_cal", "T_cal",
                "monetary_value_cal", "frequency_holdout"]
    present  = [c for c in cols_num if c in rfm.columns]

    print("\n" + "=" * 65)
    print("  RESUMEN  rfm_cal_holdout.parquet")
    print("=" * 65)
    print(f"  Clientes totales:        {len(rfm):,}")
    print(f"  Repeat buyers (cal):     "
          f"{rfm['is_repeat_buyer_cal'].sum():,}  "
          f"({100*rfm['is_repeat_buyer_cal'].mean():.1f}%)")
    print(f"  One-time buyers (cal):   "
          f"{(rfm['frequency_cal']==0).sum():,}")
    print()
    print("  Estadisticas descriptivas (columnas clave):")
    print(rfm[present].describe().round(2).to_string())
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rfm_df = build_rfm_cal_holdout()
    print_summary(rfm_df)
