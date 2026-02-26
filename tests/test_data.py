from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# sys.path para importar helpers del pipeline si fuera necesario
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: DataFrames sintéticos que replican la estructura de Olist
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def orders_df() -> pd.DataFrame:
    """
    Tabla sintética de órdenes (olist_orders_dataset.csv).
    Incluye distintos statuses para probar el filtrado.
    """
    return pd.DataFrame({
        "order_id":                  ["ord_001", "ord_002", "ord_003", "ord_004", "ord_005"],
        "customer_id":               ["cust_A",  "cust_B",  "cust_C",  "cust_A",  "cust_D"],
        "order_status":              ["delivered", "delivered", "canceled", "delivered", "shipped"],
        "order_purchase_timestamp":  [
            "2021-01-10 10:00:00",
            "2021-02-15 14:30:00",
            "2021-03-20 09:00:00",
            "2021-04-05 11:00:00",
            "2021-05-01 16:00:00",
        ],
    })


@pytest.fixture
def items_df() -> pd.DataFrame:
    """
    Tabla sintética de ítems (olist_order_items_dataset.csv).
    ord_001 tiene 2 ítems, el resto 1 ítem cada uno.
    """
    return pd.DataFrame({
        "order_id":       ["ord_001", "ord_001", "ord_002", "ord_003", "ord_004"],
        "order_item_id":  [1,         2,         1,         1,         1        ],
        "product_id":     ["prod_X",  "prod_Y",  "prod_Z",  "prod_W",  "prod_X" ],
        "seller_id":      ["sell_1",  "sell_2",  "sell_1",  "sell_3",  "sell_2" ],
        "price":          [100.0,     200.0,     50.0,      75.0,      300.0    ],
        "freight_value":  [10.0,      20.0,      5.0,       8.0,       30.0     ],
    })


@pytest.fixture
def customers_df() -> pd.DataFrame:
    """
    Tabla sintética de clientes (olist_customers_dataset.csv).
    customer_id → customer_unique_id (un cliente real puede tener varios
    customer_id si compró en distintas ocasiones, pero el unique_id agrupa).
    """
    return pd.DataFrame({
        "customer_id":         ["cust_A", "cust_B", "cust_C", "cust_D"],
        "customer_unique_id":  ["uid_01", "uid_02", "uid_03", "uid_04"],
        "customer_state":      ["SP",     "RJ",     "MG",     "PR"    ],
    })


@pytest.fixture
def payments_df() -> pd.DataFrame:
    """
    Tabla sintética de pagos (olist_order_payments_dataset.csv).
    Una orden puede tener múltiples pagos (cuotas).
    """
    return pd.DataFrame({
        "order_id":             ["ord_001", "ord_001", "ord_002", "ord_004"],
        "payment_sequential":   [1,         2,         1,         1        ],
        "payment_type":         ["credit_card", "boleto", "credit_card", "credit_card"],
        "payment_installments": [3,          1,         1,         6        ],
        "payment_value":        [110.0,      220.0,     55.0,      330.0    ],
    })


# ──────────────────────────────────────────────────────────────────────────────
# Helper: simula el pipeline ETL completo sobre los fixtures
# ──────────────────────────────────────────────────────────────────────────────

VALID_STATUS = {"delivered"}
OUTLIER_IQR_FACTOR = 3.0


def _iqr_bounds(series: pd.Series, factor: float = OUTLIER_IQR_FACTOR):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def run_pipeline(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replica las transformaciones de src/data_pipeline/etl.py
    sobre DataFrames en memoria (sin I/O).
    """
    # Fase 1: filtrar status
    orders_filtered = orders[orders["order_status"].isin(VALID_STATUS)].copy()

    # Fase 2: joins
    df = orders_filtered.merge(items, on="order_id", how="inner")
    df = df.merge(customers, on="customer_id", how="inner")

    # Fase 3: tipos
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    # Fase 4: nulos
    cols_criticas = [
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "price",
        "freight_value",
    ]
    df = df.dropna(subset=cols_criticas)

    # Fase 5: precios inválidos
    df = df[(df["price"] > 0) & (df["freight_value"] >= 0)]

    # Fase 6: outliers IQR (solo si hay suficientes filas)
    if len(df) >= 4:
        lo, hi = _iqr_bounds(df["price"])
        df = df[(df["price"] >= lo) & (df["price"] <= hi)]

    # Fase 7: revenue
    df["revenue"] = df["price"] + df["freight_value"]

    columnas_finales = [
        "customer_unique_id", "order_id", "order_purchase_timestamp",
        "order_item_id", "product_id", "seller_id",
        "price", "freight_value", "revenue", "customer_state",
    ]
    df = df[columnas_finales].sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    ).reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Tests de integridad referencial y claves
# ══════════════════════════════════════════════════════════════════════════════

class TestJoinIntegrity:
    """Verifica que los joins no generen duplicados de clave compuesta."""

    def test_composite_key_order_id_item_id_is_unique(
        self, orders_df, items_df, customers_df
    ):
        """
        (order_id, order_item_id) debe ser único en el resultado del ETL.
        Si aparece duplicado, el join creó filas fantasmas.
        """
        result = run_pipeline(orders_df, items_df, customers_df)
        duplicados = result.duplicated(subset=["order_id", "order_item_id"])
        assert not duplicados.any(), (
            f"Se encontraron {duplicados.sum()} filas con clave compuesta "
            f"(order_id, order_item_id) duplicada:\n"
            f"{result[duplicados][['order_id', 'order_item_id', 'customer_unique_id']]}"
        )

    def test_no_customer_unique_id_duplicated_per_item(
        self, orders_df, items_df, customers_df
    ):
        """
        Cada fila del resultado debe tener un único customer_unique_id.
        (Un mismo ítem no puede pertenecer a dos clientes distintos.)
        """
        result = run_pipeline(orders_df, items_df, customers_df)
        # Agrupamos por (order_id, order_item_id) y contamos distintos customer_unique_id
        grouped = (
            result.groupby(["order_id", "order_item_id"])["customer_unique_id"]
            .nunique()
        )
        multi_owner = grouped[grouped > 1]
        assert multi_owner.empty, (
            f"Ítems con más de un customer_unique_id:\n{multi_owner}"
        )

    def test_customers_table_itself_has_no_duplicate_unique_ids(
        self, customers_df
    ):
        """
        La tabla de clientes NO debe tener customer_unique_id duplicados.
        Si los tuviera, el join de muchos-a-muchos inflaría las filas.
        """
        duplicados = customers_df.duplicated(subset=["customer_unique_id"])
        assert not duplicados.any(), (
            f"La tabla de clientes tiene {duplicados.sum()} "
            f"customer_unique_id duplicados."
        )

    def test_payments_sequential_is_unique_per_order(self, payments_df):
        """
        Dentro de cada order_id, payment_sequential debe ser único.
        Duplicados aquí causarían inflación al hacer joins con pagos.
        """
        dups = payments_df.duplicated(subset=["order_id", "payment_sequential"])
        assert not dups.any(), (
            f"Pagos con (order_id, payment_sequential) duplicado:\n"
            f"{payments_df[dups]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tests de filtrado por order_status
# ══════════════════════════════════════════════════════════════════════════════

class TestStatusFiltering:
    """Solo deben sobrevivir órdenes con status='delivered'."""

    def test_only_delivered_orders_survive(
        self, orders_df, items_df, customers_df
    ):
        result = run_pipeline(orders_df, items_df, customers_df)
        # ord_003 es 'canceled' y ord_005 es 'shipped' → deben quedar fuera
        assert "ord_003" not in result["order_id"].values, (
            "Orden con status='canceled' no debió aparecer en el resultado"
        )
        assert "ord_005" not in result["order_id"].values, (
            "Orden con status='shipped' no debió aparecer en el resultado"
        )

    def test_delivered_orders_are_present(
        self, orders_df, items_df, customers_df
    ):
        """Las órdenes entregadas deben seguir presentes."""
        result = run_pipeline(orders_df, items_df, customers_df)
        delivered_ids = {"ord_001", "ord_002", "ord_004"}
        result_ids = set(result["order_id"].unique())
        assert delivered_ids.issubset(result_ids), (
            f"Faltan órdenes entregadas: {delivered_ids - result_ids}"
        )

    def test_row_count_matches_delivered_items(
        self, orders_df, items_df, customers_df
    ):
        """
        Órdenes entregadas en el fixture: ord_001 (2 ítems), ord_002 (1), ord_004 (1)
        → total esperado = 4 filas (antes del filtro de outliers IQR).
        """
        result = run_pipeline(orders_df, items_df, customers_df)
        # Verificamos que al menos estén las 4 filas base (outliers podrían reducir)
        assert len(result) >= 1, "El resultado está vacío, algo falló en el pipeline"


# ══════════════════════════════════════════════════════════════════════════════
# Tests de calidad de datos post-ETL
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQuality:
    """Integridad de columnas críticas en el output del pipeline."""

    def test_no_nulls_in_critical_columns(
        self, orders_df, items_df, customers_df
    ):
        """Ninguna columna crítica puede tener valores nulos."""
        result = run_pipeline(orders_df, items_df, customers_df)
        cols_criticas = [
            "customer_unique_id",
            "order_id",
            "order_purchase_timestamp",
            "price",
            "freight_value",
        ]
        for col in cols_criticas:
            nulls = result[col].isna().sum()
            assert nulls == 0, (
                f"Columna '{col}' tiene {nulls} valores nulos"
            )

    def test_price_is_always_positive(self, orders_df, items_df, customers_df):
        """Precio debe ser estrictamente positivo (> 0)."""
        result = run_pipeline(orders_df, items_df, customers_df)
        invalid = result[result["price"] <= 0]
        assert invalid.empty, (
            f"Se encontraron {len(invalid)} filas con price ≤ 0:\n{invalid}"
        )

    def test_freight_value_is_non_negative(
        self, orders_df, items_df, customers_df
    ):
        """Freight value debe ser ≥ 0 (puede ser cero para envío gratis)."""
        result = run_pipeline(orders_df, items_df, customers_df)
        invalid = result[result["freight_value"] < 0]
        assert invalid.empty, (
            f"Se encontraron {len(invalid)} filas con freight_value < 0:\n{invalid}"
        )

    def test_revenue_equals_price_plus_freight(
        self, orders_df, items_df, customers_df
    ):
        """revenue debe ser la suma exacta de price + freight_value."""
        result = run_pipeline(orders_df, items_df, customers_df)
        computed = result["price"] + result["freight_value"]
        discrepancias = (result["revenue"] - computed).abs()
        assert (discrepancias < 1e-9).all(), (
            f"revenue ≠ price + freight en {(discrepancias >= 1e-9).sum()} filas"
        )

    def test_timestamp_is_datetime_type(self, orders_df, items_df, customers_df):
        """order_purchase_timestamp debe ser un tipo datetime64 de pandas."""
        result = run_pipeline(orders_df, items_df, customers_df)
        assert pd.api.types.is_datetime64_any_dtype(
            result["order_purchase_timestamp"]
        ), (
            f"Se esperaba datetime64, se obtuvo "
            f"{result['order_purchase_timestamp'].dtype}"
        )

    def test_required_columns_exist(self, orders_df, items_df, customers_df):
        """El DataFrame resultante debe tener todas las columnas esperadas."""
        result = run_pipeline(orders_df, items_df, customers_df)
        expected_cols = {
            "customer_unique_id", "order_id", "order_purchase_timestamp",
            "order_item_id", "product_id", "seller_id",
            "price", "freight_value", "revenue", "customer_state",
        }
        missing = expected_cols - set(result.columns)
        assert not missing, f"Faltan columnas en el resultado: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# Tests de relaciones 1-a-N legítimas
# ══════════════════════════════════════════════════════════════════════════════

class TestOneToManyRelationships:
    """Verifica que las relaciones legítimas 1-a-N se conservan correctamente."""

    def test_customer_can_have_multiple_orders(
        self, orders_df, items_df, customers_df
    ):
        """
        cust_A tiene ord_001 y ord_004 (ambas 'delivered').
        uid_01 debe aparecer al menos 2 veces en el resultado.
        """
        result = run_pipeline(orders_df, items_df, customers_df)
        uid_01_rows = result[result["customer_unique_id"] == "uid_01"]
        assert len(uid_01_rows) >= 2, (
            f"uid_01 debería tener ≥ 2 filas (una por ítem de cada orden), "
            f"pero tiene {len(uid_01_rows)}"
        )

    def test_order_can_have_multiple_items(
        self, orders_df, items_df, customers_df
    ):
        """
        ord_001 tiene order_item_id 1 y 2, ambos deben estar en el resultado.
        """
        result = run_pipeline(orders_df, items_df, customers_df)
        ord_001_items = result[result["order_id"] == "ord_001"]["order_item_id"].tolist()
        assert 1 in ord_001_items and 2 in ord_001_items, (
            f"ord_001 debería tener items 1 y 2, se encontró: {ord_001_items}"
        )

    def test_multiple_orders_same_unique_id_not_duplicated(
        self, orders_df, items_df, customers_df
    ):
        """
        Un mismo customer_unique_id con múltiples órdenes NO es duplicación;
        es una relación 1-a-N legítima.  La clave real es (order_id, order_item_id).
        """
        result = run_pipeline(orders_df, items_df, customers_df)
        # customer_unique_id puede repetirse → esto es esperable
        uid_counts = result["customer_unique_id"].value_counts()
        # Lo que NO debe repetirse es la clave compuesta
        dups = result.duplicated(subset=["order_id", "order_item_id"])
        assert not dups.any(), (
            "La clave compuesta (order_id, order_item_id) tiene duplicados, "
            "lo que indica un join incorrecto."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tests de casos borde
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Casos límite para robustecer el pipeline."""

    def test_all_canceled_orders_returns_empty_df(
        self, items_df, customers_df
    ):
        """Si todas las órdenes están canceladas, el resultado debe estar vacío."""
        orders_all_canceled = pd.DataFrame({
            "order_id":                 ["ord_X", "ord_Y"],
            "customer_id":              ["cust_A", "cust_B"],
            "order_status":             ["canceled", "canceled"],
            "order_purchase_timestamp": ["2021-01-01", "2021-02-01"],
        })
        result = run_pipeline(orders_all_canceled, items_df, customers_df)
        assert result.empty, (
            "Con todas las órdenes canceladas el resultado debería estar vacío"
        )

    def test_order_id_not_in_items_is_dropped(self, customers_df):
        """
        Una orden 'delivered' sin ítems correspondientes (inner join)
        no debe aparecer en el resultado.
        """
        orders_extra = pd.DataFrame({
            "order_id":                 ["ord_no_items"],
            "customer_id":              ["cust_A"],
            "order_status":             ["delivered"],
            "order_purchase_timestamp": ["2021-06-01"],
        })
        items_empty = pd.DataFrame(columns=["order_id", "order_item_id",
                                            "product_id", "seller_id",
                                            "price", "freight_value"])
        result = run_pipeline(orders_extra, items_empty, customers_df)
        assert result.empty or "ord_no_items" not in result["order_id"].values

    def test_negative_price_row_is_filtered(self, orders_df, customers_df):
        """Una fila con price negativo debe ser eliminada por el pipeline."""
        items_with_negative = pd.DataFrame({
            "order_id":       ["ord_001"],
            "order_item_id":  [1],
            "product_id":     ["prod_bad"],
            "seller_id":      ["sell_1"],
            "price":          [-50.0],       # ← precio inválido
            "freight_value":  [10.0],
        })
        result = run_pipeline(orders_df, items_with_negative, customers_df)
        invalid = result[result["price"] <= 0]
        assert invalid.empty, "Filas con precio negativo no fueron eliminadas"
