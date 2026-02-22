"""
monitor.py — Monitoreo de Deriva de Datos (Data Drift) con Evidently AI
========================================================================
Compara la distribución de los datos RFM con los que recibe la API en
producción. Si el patrón de gasto de los clientes cambia significativamente,
el reporte lo señala en rojo.

Uso
---
    # Modo normal: carga reference de rfm_clv.parquet y current de SQLite
    python -m src.monitoring.monitor

    # Modo demo: genera datos de producción sintéticos para probar el reporte
    python -m src.monitoring.monitor --demo

    # Especificar ruta de salida del reporte HTML
    python -m src.monitoring.monitor --output reports/drift/mi_reporte.html

    # Mínimo de filas en producción para correr el análisis (default: 30)
    python -m src.monitoring.monitor --min-rows 50

Salida
------
    reports/drift/drift_report_<TIMESTAMP>.html   ← reporte visual interactivo
    reports/drift/drift_summary_<TIMESTAMP>.json  ← resumen en JSON (métricas)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Evidently v0.7+ ────────────────────────────────────────────────────────
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

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
# Rutas base
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
REFERENCE_PATH  = ROOT / "data" / "processed"  / "rfm_clv.parquet"
PROD_DB_PATH    = ROOT / "data" / "production_logs" / "requests.db"
PROD_CSV_PATH   = ROOT / "data" / "production_logs" / "requests.csv"
REPORTS_DIR     = ROOT / "reports" / "drift"

# Columnas RFM que interesa monitorear
RFM_COLS = ["frequency", "recency", "T", "monetary_value"]


# ===========================================================================
# 1. CARGA DE DATOS DE REFERENCIA (entrenamiento)
# ===========================================================================

def load_reference() -> pd.DataFrame:
    """
    Carga la tabla RFM de entrenamiento desde rfm_clv.parquet.
    Solo extrae las 4 columnas RFM core para hacer la comparación.
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de referencia en: {REFERENCE_PATH}\n"
            "Asegurate de haber corrido el pipeline ETL y el modelado RFM."
        )

    log.info("Cargando datos de referencia desde: %s", REFERENCE_PATH)
    df = pd.read_parquet(REFERENCE_PATH, columns=RFM_COLS)
    log.info("  → %d filas de referencia cargadas.", len(df))
    return df.reset_index(drop=True)


# ===========================================================================
# 2. CARGA DE LOGS DE PRODUCCIÓN
# ===========================================================================

def load_production_from_sqlite() -> pd.DataFrame | None:
    """
    Lee los requests registrados por FastAPI desde la base SQLite.
    Devuelve None si la base no existe o está vacía.
    """
    if not PROD_DB_PATH.exists():
        log.warning("Base SQLite no encontrada en: %s", PROD_DB_PATH)
        return None

    try:
        con = sqlite3.connect(str(PROD_DB_PATH))
        df = pd.read_sql(
            "SELECT frequency, recency, T, monetary_value FROM prediction_logs",
            con,
        )
        con.close()
        log.info("  → %d filas de producción leídas desde SQLite.", len(df))
        return df if not df.empty else None
    except Exception as exc:
        log.warning("No se pudo leer la base SQLite: %s", exc)
        return None


def load_production_from_csv() -> pd.DataFrame | None:
    """
    Fallback: lee los logs desde el CSV plano.
    """
    if not PROD_CSV_PATH.exists():
        log.warning("CSV de producción no encontrado en: %s", PROD_CSV_PATH)
        return None

    try:
        df = pd.read_csv(PROD_CSV_PATH, usecols=RFM_COLS)
        log.info("  → %d filas de producción leídas desde CSV.", len(df))
        return df if not df.empty else None
    except Exception as exc:
        log.warning("No se pudo leer el CSV de producción: %s", exc)
        return None


def load_production() -> pd.DataFrame | None:
    """
    Intenta cargar producción primero desde SQLite y luego desde CSV.
    """
    log.info("Cargando datos de producción…")
    df = load_production_from_sqlite()
    if df is None:
        df = load_production_from_csv()
    return df


# ===========================================================================
# 3. DATOS SINTÉTICOS DE DEMO
# ===========================================================================

def make_demo_production(reference: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """
    Genera datos de producción sintéticos con deriva simulada:
      - frequency:      igual que referencia (sin deriva)
      - recency:        ligeramente aumentada (+20 días de media)
      - T:              igual que referencia (sin deriva)
      - monetary_value: DUPLICADA → deriva fuerte (alerta roja en el reporte)

    Esto permite demostrar visualmente el sistema sin necesidad de tener
    la API corriendo en producción.
    """
    rng = np.random.default_rng(seed=42)

    # Tomamos una muestra de la referencia como base
    sample = reference.sample(n=min(n, len(reference)), random_state=42)

    demo = pd.DataFrame({
        "frequency":      sample["frequency"].values,
        "recency":        sample["recency"].values + rng.normal(20, 10, n).clip(0),
        "T":              sample["T"].values,
        # Deriva fuerte en monetary_value: simulamos un segmento premium
        "monetary_value": sample["monetary_value"].values * rng.uniform(1.8, 2.5, n),
    })
    log.info("Modo DEMO: %d filas de producción sintéticas generadas (deriva en monetary_value).", n)
    return demo.reset_index(drop=True)


# ===========================================================================
# 4. GENERACIÓN DEL REPORTE EVIDENTLY
# ===========================================================================

def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> dict:
    """
    Ejecuta el análisis de deriva con Evidently y guarda el reporte HTML.

    Utiliza:
      - DataDriftPreset  → detecta si la distribución de cada columna
                           cambió significativamente (test estadístico automático)
      - DataSummaryPreset→ estadísticas descriptivas de referencia vs producción

    Devuelve un diccionario con las métricas clave para el resumen JSON.
    """
    log.info("Construyendo datasets Evidently…")
    ds_ref = Dataset.from_pandas(reference)
    ds_cur = Dataset.from_pandas(current)

    log.info("Ejecutando análisis de deriva (DataDriftPreset + DataSummaryPreset)…")
    # En Evidently v0.7, Report([presets]).run() devuelve un Snapshot
    report_def = Report([
        DataDriftPreset(),
        DataSummaryPreset(),
    ])
    snapshot = report_def.run(
        reference_data=ds_ref,
        current_data=ds_cur,
    )

    # Guardar reporte HTML interactivo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))
    log.info("Reporte HTML guardado en: %s", output_path)

    # Extraer métricas para el JSON de resumen
    summary = _extract_summary(snapshot, reference, current)
    return summary


def _extract_summary(
    snapshot,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """
    Extrae un resumen de las diferencias estadísticas entre reference y current
    para guardarlo como JSON legible por máquinas.
    """
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "columns_analyzed": RFM_COLS,
        "column_stats": {},
        "drift_detected": {},
    }

    for col in RFM_COLS:
        ref_col = reference[col].dropna()
        cur_col = current[col].dropna()

        # Diferencia de medias en %
        ref_mean = ref_col.mean()
        cur_mean = cur_col.mean()
        pct_change = ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0.0

        # Test de Kolmogorov-Smirnov básico (disponible en scipy vía pandas describe)
        from scipy import stats as sp_stats  # noqa: PLC0415
        ks_stat, ks_pvalue = sp_stats.ks_2samp(ref_col.values, cur_col.values)

        drift_flag = bool(ks_pvalue < 0.05)

        summary["column_stats"][col] = {
            "ref_mean":    round(float(ref_mean), 4),
            "cur_mean":    round(float(cur_mean), 4),
            "pct_change":  round(float(pct_change), 2),
            "ks_stat":     round(float(ks_stat), 4),
            "ks_pvalue":   round(float(ks_pvalue), 6),
        }
        summary["drift_detected"][col] = drift_flag

    # Flag global: hay deriva si al menos 1 columna tiene deriva
    summary["any_drift_detected"] = any(summary["drift_detected"].values())

    return summary


def _print_summary(summary: dict) -> None:
    """Imprime el resumen de deriva en la consola con formato visual claro."""
    print("\n" + "=" * 64)
    print("  RESUMEN DE DERIVA -- CLV Forecasting API")
    print("=" * 64)
    print(f"  Filas referencia : {summary['reference_rows']:,}")
    print(f"  Filas produccion : {summary['current_rows']:,}")
    print(f"  Generado en      : {summary['generated_at']}")
    print("-" * 64)
    print(f"  {'Columna':<20} {'Media Ref':>10} {'Media Act':>10} {'Dif%':>8}  {'Deriva':>12}")
    print("-" * 64)

    for col in RFM_COLS:
        stats = summary["column_stats"][col]
        drift = summary["drift_detected"][col]
        icon  = "[DERIVA]" if drift else "[  OK  ]"
        print(
            f"  {col:<20} "
            f"{stats['ref_mean']:>10.3f} "
            f"{stats['cur_mean']:>10.3f} "
            f"{stats['pct_change']:>7.1f}%  "
            f"{icon:>12}"
        )

    print("=" * 64)
    global_flag = summary["any_drift_detected"]
    if global_flag:
        print("  ** ALERTA: Se detecto deriva en al menos una variable RFM. **")
        print("  Revisa el reporte HTML para ver los detalles por columna.")
    else:
        print("  OK: Sin deriva detectada. Las distribuciones son estables.")
    print("=" * 64 + "\n")


# ===========================================================================
# 5. ENTRY POINT
# ===========================================================================

def run_monitor(
    demo: bool = False,
    output: Path | None = None,
    min_rows: int = 30,
) -> int:
    """
    Lógica principal del monitor. Devuelve 0 si OK, 1 si hubo error.

    Parámetros
    ----------
    demo     : si True, genera datos de producción sintéticos con deriva
    output   : ruta del HTML de salida (None = auto con timestamp)
    min_rows : mínimo de filas de producción para correr el análisis
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output is None:
        output = REPORTS_DIR / f"drift_report_{timestamp}.html"

    json_output = output.with_name(output.stem.replace("drift_report", "drift_summary") + ".json")

    # ── 1. Referencia ───────────────────────────────────────────────────────
    try:
        reference = load_reference()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    # ── 2. Producción ───────────────────────────────────────────────────────
    if demo:
        current = make_demo_production(reference)
    else:
        current = load_production()
        if current is None:
            log.error(
                "No hay datos de producción disponibles.\n"
                "  → Asegurate de que la API FastAPI esté corriendo y haya\n"
                "    recibido al menos %d requests.\n"
                "  → Alternativamente, corré con --demo para ver un ejemplo.",
                min_rows,
            )
            return 1

        if len(current) < min_rows:
            log.error(
                "Solo hay %d filas de producción (mínimo requerido: %d).\n"
                "  → Usá --demo para generar datos sintéticos de prueba.",
                len(current), min_rows,
            )
            return 1

    # ── 3. Reporte ──────────────────────────────────────────────────────────
    try:
        summary = build_report(reference, current, output)
    except Exception as exc:
        log.exception("Error al generar el reporte Evidently: %s", exc)
        return 1

    # ── 4. Resumen en consola ───────────────────────────────────────────────
    _print_summary(summary)

    # ── 5. Guardar JSON de métricas ─────────────────────────────────────────
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Resumen JSON guardado en: %s", json_output)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor de deriva de datos RFM para la API CLV Forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Genera datos de producción sintéticos con deriva para demostración.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Ruta del reporte HTML de salida. Por defecto: reports/drift/drift_report_<timestamp>.html",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=30,
        metavar="N",
        help="Mínimo de filas de producción para correr el análisis (default: 30).",
    )
    args = parser.parse_args()

    exit_code = run_monitor(
        demo=args.demo,
        output=args.output,
        min_rows=args.min_rows,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
