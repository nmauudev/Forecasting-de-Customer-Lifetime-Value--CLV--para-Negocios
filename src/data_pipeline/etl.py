from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]          # project root
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "clean_transactions.parquet"


VALID_STATUS = {"delivered"}                        # solo compras reales


OUTLIER_IQR_FACTOR = 3.0



def _load_csv(filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Carga un CSV desde RAW_DIR con logging del shape."""
    path = RAW_DIR / filename
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    log.info("Loaded %-45s  shape=%s", filename, df.shape)
    return df


def _iqr_bounds(series: pd.Series, factor: float = OUTLIER_IQR_FACTOR):
    """Devuelve (lower, upper) basado en IQR * factor."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr




def run_etl() -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo y devuelve el DataFrame limpio.
    También guarda el resultado en OUTPUT_FILE.
    """

    
    log.info("=== FASE 1: Extracción ===")

    orders = _load_csv(
        "olist_orders_dataset.csv",
        usecols=[
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ],
    )

    items = _load_csv(
        "olist_order_items_dataset.csv",
        usecols=[
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ],
    )

    customers = _load_csv(
        "olist_customers_dataset.csv",
        usecols=[
            "customer_id",
            "customer_unique_id",
            "customer_state",
        ],
    )

    
    log.info("=== FASE 2: Filtro de status ===")
    log.info("Órdenes totales antes del filtro: %d", len(orders))

    orders = orders[orders["order_status"].isin(VALID_STATUS)].copy()
    log.info("Órdenes tras filtrar status='delivered': %d", len(orders))

    
    log.info("=== FASE 3: Joins ===")

    
    df = orders.merge(items, on="order_id", how="inner")
    log.info("Tras join orders+items:     shape=%s", df.shape)

    
    df = df.merge(customers, on="customer_id", how="inner")
    log.info("Tras join +customers:       shape=%s", df.shape)

    
    log.info("=== FASE 4: Tipos ===")

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    
    log.info("=== FASE 5: Limpieza de nulos ===")
    cols_criticas = [
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "price",
        "freight_value",
    ]
    antes = len(df)
    df = df.dropna(subset=cols_criticas)
    log.info(
        "Filas eliminadas por nulos en columnas críticas: %d  (quedan %d)",
        antes - len(df),
        len(df),
    )

    
    log.info("=== FASE 6: Valores inválidos en price/freight ===")
    antes = len(df)
    df = df[(df["price"] > 0) & (df["freight_value"] >= 0)]
    log.info(
        "Filas eliminadas por price≤0 o freight<0: %d  (quedan %d)",
        antes - len(df),
        len(df),
    )

    
    log.info("=== FASE 7: Outliers en price (IQR ×%.1f) ===", OUTLIER_IQR_FACTOR)
    lo, hi = _iqr_bounds(df["price"])
    antes = len(df)
    df = df[(df["price"] >= lo) & (df["price"] <= hi)]
    log.info(
        "Rango precio aceptado: [%.2f, %.2f]  →  Filas eliminadas: %d  (quedan %d)",
        lo, hi, antes - len(df), len(df),
    )

    
    df["revenue"] = df["price"] + df["freight_value"]

    
    columnas_finales = [
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
        "revenue",
        "customer_state",
    ]
    df = df[columnas_finales].sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    ).reset_index(drop=True)

    log.info("=== Shape final del DataFrame limpio: %s ===", df.shape)

    
    log.info("=== FASE 8: Guardando parquet ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    log.info("Archivo guardado en: %s", OUTPUT_FILE)

    return df




def print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen ejecutivo del DataFrame resultante."""
    print("\n" + "=" * 60)
    print("  RESUMEN  clean_transactions.parquet")
    print("=" * 60)
    print(f"  Filas (ítems de transacción):  {len(df):,}")
    print(f"  Órdenes únicas:                {df['order_id'].nunique():,}")
    print(f"  Clientes únicos:               {df['customer_unique_id'].nunique():,}")
    print(f"  Rango fechas: {df['order_purchase_timestamp'].min().date()} "
          f"-> {df['order_purchase_timestamp'].max().date()}")
    print(f"  Revenue total:                 ${df['revenue'].sum():,.2f}")
    print(f"  Price  media:                  ${df['price'].mean():.2f}  "
          f"(std={df['price'].std():.2f})")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    clean_df = run_etl()
    print_summary(clean_df)
