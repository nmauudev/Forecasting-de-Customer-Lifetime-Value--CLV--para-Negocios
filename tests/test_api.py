from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Ajuste del sys.path para que la importación de src.api.main funcione
# independientemente del directorio de trabajo.
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: mocks de modelos y tabla RFM
# ──────────────────────────────────────────────────────────────────────────────

def _make_bg_nbd_mock() -> MagicMock:
    """Crea un mock del modelo BG/NBD con valores deterministas."""
    mock = MagicMock()
    # predict → compras esperadas = 2.5
    mock.predict.return_value = pd.Series([2.5])
    # prob de estar vivo = 0.85
    mock.conditional_probability_alive.return_value = pd.Series([0.85])
    return mock


def _make_gamma_gamma_mock() -> MagicMock:
    """Crea un mock del modelo Gamma-Gamma con revenue esperado = 180.0."""
    mock = MagicMock()
    mock.conditional_expected_average_profit.return_value = pd.Series([180.0])
    return mock


def _make_rfm_df() -> pd.DataFrame:
    """
    Crea una tabla RFM mínima con 3 clientes ficticios.
    El índice es customer_unique_id, igual que en producción.
    """
    data = {
        "frequency":           [3.0,  1.0,  5.0],
        "recency":             [120.0, 30.0, 200.0],
        "T":                   [365.0, 90.0, 500.0],
        "monetary_value":      [150.0, 80.0, 300.0],
        "clv_12m":             [450.0, 80.0, 1500.0],
        "prob_activo":         [0.9,   0.7,  0.95],
    }
    ids = ["cliente_aaa", "cliente_bbb", "cliente_ccc"]
    return pd.DataFrame(data, index=pd.Index(ids, name="customer_unique_id"))


# ──────────────────────────────────────────────────────────────────────────────
# Configuración de los patches antes de importar main.py
# (joblib.load y pd.read_parquet se interceptan para no tocar el disco)
# ──────────────────────────────────────────────────────────────────────────────

BG_MOCK        = _make_bg_nbd_mock()
GG_MOCK        = _make_gamma_gamma_mock()
RFM_DF_MOCK    = _make_rfm_df()


def _joblib_side_effect(path, *args, **kwargs):
    """Devuelve el mock correcto según el nombre del archivo."""
    path_str = str(path)
    if "bg_nbd" in path_str:
        return BG_MOCK
    if "gamma_gamma" in path_str:
        return GG_MOCK
    raise FileNotFoundError(f"Mock no configurado para: {path}")


with (
    patch("joblib.load", side_effect=_joblib_side_effect),
    patch("pandas.read_parquet", return_value=RFM_DF_MOCK),
):
    from src.api.main import app          # noqa: E402

from starlette.testclient import TestClient   # noqa: E402  (después del patch)

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Endpoint /health – debe responder siempre status 200 con models_loaded."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Se esperaba 200 pero se obtuvo {response.status_code}"
        )

    def test_health_body_has_status_ok(self):
        response = client.get("/health")
        body = response.json()
        assert body.get("status") == "ok"

    def test_health_body_models_loaded_true(self):
        response = client.get("/health")
        body = response.json()
        assert body.get("models_loaded") is True


class TestPredictCLVEndpoint:
    """Endpoint POST /predict-clv – verificamos CLV, tipos y manejo de errores."""

    # ── Payload base reutilizable ─────────────────────────────────────────────
    VALID_PAYLOAD = {
        "frequency":      3.0,
        "recency":        120.0,
        "T":              365.0,
        "monetary_value": 150.0,
        "months":         12,
    }

    def test_valid_payload_returns_200(self):
        response = client.post("/predict-clv", json=self.VALID_PAYLOAD)
        assert response.status_code == 200, (
            f"Se esperaba 200 pero se obtuvo {response.status_code}.\n"
            f"Body: {response.text}"
        )

    def test_clv_predicted_is_positive_number(self):
        """CLV = expected_purchases × expected_avg_revenue → debe ser > 0."""
        response = client.post("/predict-clv", json=self.VALID_PAYLOAD)
        body = response.json()
        clv = body["clv_predicted"]
        assert isinstance(clv, (int, float)), "clv_predicted debe ser numérico"
        assert clv > 0, f"CLV debe ser positivo, se obtuvo {clv}"

    def test_clv_value_matches_expected(self):
        """
        Con los mocks: expected_purchases=2.5, expected_avg_revenue=180.0
        → CLV esperado = 2.5 × 180.0 = 450.0
        """
        response = client.post("/predict-clv", json=self.VALID_PAYLOAD)
        body = response.json()
        assert body["clv_predicted"] == pytest.approx(450.0, rel=1e-3), (
            f"CLV incorrecto: {body['clv_predicted']} (esperado ~450.0)"
        )

    def test_response_contains_all_required_fields(self):
        """El schema CLVOutput debe tener exactamente los 5 campos definidos."""
        required_fields = {
            "clv_predicted",
            "expected_purchases",
            "expected_avg_revenue",
            "prob_alive",
            "horizon_months",
        }
        response = client.post("/predict-clv", json=self.VALID_PAYLOAD)
        body = response.json()
        assert required_fields.issubset(body.keys()), (
            f"Faltan campos en la respuesta: {required_fields - body.keys()}"
        )

    def test_horizon_months_echoes_input(self):
        """El campo horizon_months debe reflejar el valor enviado."""
        payload = {**self.VALID_PAYLOAD, "months": 6}
        response = client.post("/predict-clv", json=payload)
        assert response.json()["horizon_months"] == 6

    def test_frequency_zero_returns_200_not_500(self):
        """
        Si frequency=0, el modelo Gamma-Gamma no se llama (rama else).
        No debe devolver 500.
        """
        payload = {
            "frequency":      0.0,
            "recency":        0.0,
            "T":              30.0,
            "monetary_value": 0.0,
            "months":         12,
        }
        response = client.post("/predict-clv", json=payload)
        assert response.status_code == 200, (
            f"frequency=0 generó {response.status_code}, esperado 200.\n"
            f"Body: {response.text}"
        )
        # Con freq=0 → monetary_value=0, CLV debe ser ≥ 0 (no negativo)
        assert response.json()["clv_predicted"] >= 0

    def test_invalid_T_zero_returns_422_not_500(self):
        """T=0 viola la restricción gt=0 del schema → Pydantic lanza 422."""
        payload = {**self.VALID_PAYLOAD, "T": 0.0}
        response = client.post("/predict-clv", json=payload)
        assert response.status_code == 422, (
            f"T=0 debería retornar 422 (validación), no {response.status_code}"
        )
        # Verificar que NO es un error interno del servidor
        assert response.status_code != 500, "El API devolvió un error 500 inesperado"

    def test_negative_frequency_returns_422(self):
        """frequency=-1 viola la restricción ge=0 → 422."""
        payload = {**self.VALID_PAYLOAD, "frequency": -1.0}
        response = client.post("/predict-clv", json=payload)
        assert response.status_code == 422

    def test_boundary_high_values_dont_crash(self):
        """Valores extremos pero válidos no deben provocar un 500."""
        payload = {
            "frequency":      999.0,
            "recency":        9999.0,
            "T":              10000.0,
            "monetary_value": 99999.99,
            "months":         24,
        }
        response = client.post("/predict-clv", json=payload)
        # No debe ser un 500 (el modelo mock siempre devuelve valores fijos)
        assert response.status_code != 500, (
            f"Valores extremos causaron error 500.\nBody: {response.text}"
        )

    def test_prob_alive_between_0_and_1(self):
        """La probabilidad de estar vivo debe estar en [0, 1]."""
        response = client.post("/predict-clv", json=self.VALID_PAYLOAD)
        prob = response.json()["prob_alive"]
        assert 0.0 <= prob <= 1.0, f"prob_alive fuera de rango: {prob}"

    def test_missing_required_field_returns_422(self):
        """Si falta un campo obligatorio (monetary_value), debe ser 422."""
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "monetary_value"}
        response = client.post("/predict-clv", json=payload)
        assert response.status_code == 422


class TestCustomerEndpoint:
    """Endpoint GET /customer/{id} – lookup en la tabla RFM."""

    def test_known_customer_returns_200(self):
        response = client.get("/customer/cliente_aaa")
        assert response.status_code == 200, (
            f"Se esperaba 200 pero se obtuvo {response.status_code}.\n"
            f"Body: {response.text}"
        )

    def test_known_customer_returns_correct_rfm(self):
        response = client.get("/customer/cliente_aaa")
        body = response.json()
        assert body["frequency"]       == pytest.approx(3.0)
        assert body["recency"]         == pytest.approx(120.0)
        assert body["T"]               == pytest.approx(365.0)
        assert body["monetary_value"]  == pytest.approx(150.0)
        assert body["clv_12m_precomputed"] == pytest.approx(450.0)

    def test_known_customer_id_echoed(self):
        response = client.get("/customer/cliente_bbb")
        assert response.json()["customer_unique_id"] == "cliente_bbb"

    def test_unknown_customer_returns_404(self):
        response = client.get("/customer/id_que_no_existe_xyz")
        assert response.status_code == 404, (
            f"Se esperaba 404 pero se obtuvo {response.status_code}"
        )

    def test_unknown_customer_detail_message(self):
        response = client.get("/customer/id_fantasma")
        body = response.json()
        assert "detail" in body
        assert "id_fantasma" in body["detail"]
