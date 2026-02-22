"""
CLV Forecasting — Interfaz Streamlit
======================================
Levanta en el puerto 8501.
Conecta con el backend FastAPI en http://api:8000 (Docker) o http://localhost:8000 (local).
"""

import os
import time

import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Forecasting · Olist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# URL del backend (env var o fallback a localhost)
# ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ──────────────────────────────────────────────
# CSS personalizado — look premium oscuro
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo principal */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards métricas */
    .clv-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .clv-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.25);
    }
    .clv-card .label {
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 0.4rem;
    }
    .clv-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
    }
    .clv-card .unit {
        font-size: 0.85rem;
        color: #818cf8;
        margin-top: 0.2rem;
    }

    /* Botón principal */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.03em;
        transition: opacity 0.2s ease, transform 0.15s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    /* Título hero */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a5b4fc, #818cf8, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Separador */
    hr { border-color: rgba(255,255,255,0.08); }

    /* Sliders y inputs */
    .stSlider > div > div > div { background: #6366f1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def fetch_customer(customer_id: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/customer/{customer_id.strip()}", timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def predict_clv(frequency: float, recency: float, T: float, monetary_value: float, months: int = 12) -> dict | None:
    payload = {
        "frequency":      frequency,
        "recency":        recency,
        "T":              T,
        "monetary_value": monetary_value,
        "months":         months,
    }
    try:
        r = requests.post(f"{API_URL}/predict-clv", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"Error del servidor: {r.status_code} — {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ No se puede conectar con el backend. ¿Está corriendo en el puerto 8000?")
        return None
    except Exception as exc:
        st.error(f"Error inesperado: {exc}")
        return None


def get_sample_customers() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/customers/sample?n=20", timeout=5)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


# ──────────────────────────────────────────────
# Función para generar el gráfico de gauge CLV
# ──────────────────────────────────────────────

def make_gauge_chart(value: float, max_val: float = 500.0) -> go.Figure:
    pct = min(value / max_val, 1.0)

    if pct < 0.33:
        bar_color = "#f87171"   # rojo suave
    elif pct < 0.66:
        bar_color = "#fbbf24"   # amarillo
    else:
        bar_color = "#34d399"   # verde

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": max_val * 0.33, "valueformat": ".2f", "prefix": "vs P33: $"},
        number={"prefix": "$", "valueformat": ".2f", "font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickcolor": "#94a3b8",
                "tickfont": {"color": "#94a3b8"},
            },
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0,         max_val * 0.33], "color": "rgba(248,113,113,0.15)"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "rgba(251,191,36,0.15)"},
                {"range": [max_val * 0.66, max_val],   "color": "rgba(52,211,153,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
        title={"text": "CLV 12 Meses", "font": {"size": 16, "color": "#a5b4fc"}},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=300,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


def make_horizon_bar_chart(results: list[dict]) -> go.Figure:
    """Barra horizontal mostrando CLV para distintos horizontes."""
    horizons = [r["horizon_months"] for r in results]
    clvs     = [r["clv_predicted"]  for r in results]

    fig = go.Figure(go.Bar(
        x=clvs,
        y=[f"{h}m" for h in horizons],
        orientation="h",
        marker=dict(
            color=clvs,
            colorscale=[[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#34d399"]],
            showscale=False,
        ),
        text=[f"${c:,.2f}" for c in clvs],
        textposition="outside",
        textfont={"color": "white", "size": 13},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        title=dict(text="CLV por Horizonte de Tiempo", font={"color": "#a5b4fc", "size": 15}),
        height=250,
        margin=dict(l=40, r=80, t=50, b=20),
    )
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel de Control")
    st.divider()

    # Estado del backend
    api_ok = check_api_health()
    if api_ok:
        st.success("🟢 Backend API · Conectado", icon=None)
    else:
        st.error("🔴 Backend API · Sin conexión")
    st.caption(f"URL: `{API_URL}`")
    st.divider()

    # Modo de entrada
    mode = st.radio(
        "Modo de predicción",
        options=["🔍 Buscar cliente por ID", "🎛️ Ingresar RFM manualmente"],
        help="Elegí si querés buscar un cliente de la base Olist o ingresar los valores a mano.",
    )
    st.divider()

    # Horizonte extra
    st.markdown("### 📅 Horizontes adicionales")
    extra_horizons = st.multiselect(
        "Comparar con otros horizontes",
        options=[3, 6, 18, 24, 36],
        default=[6, 24],
        help="Se calculará el CLV para cada horizonte seleccionado.",
    )

    st.divider()
    st.caption("CLV Forecasting · BG/NBD + Gamma-Gamma  \n© 2025 · Proyecto Olist")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">📊 CLV Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Predecí el Customer Lifetime Value a 12 meses usando modelos BG/NBD y Gamma-Gamma entrenados sobre datos de Olist.</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Estado de sesión por defecto ──
defaults = {"frequency": 1.0, "recency": 90.0, "T": 365.0, "monetary_value": 150.0}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# MODO 1: Buscar cliente por ID
# ─────────────────────────────────────────────
if "Buscar" in mode:
    col_id, col_btn = st.columns([4, 1])
    with col_id:
        customer_id = st.text_input(
            "Customer Unique ID",
            placeholder="Ej: 0000366f3b9a7992bf8c76cfdf3221e2",
            label_visibility="collapsed",
        )
    with col_btn:
        buscar = st.button("Buscar", key="btn_buscar")

    # Muestra de IDs disponibles
    with st.expander("🔎 Ver muestra de clientes disponibles"):
        if st.button("Cargar muestra", key="btn_sample"):
            sample = get_sample_customers()
            if sample:
                import pandas as pd
                sample_df = pd.DataFrame(sample)
                sample_df.columns = ["Customer ID", "CLV 12m (pre-comp.)", "Frecuencia"]
                sample_df["CLV 12m (pre-comp.)"] = sample_df["CLV 12m (pre-comp.)"].map("${:,.2f}".format)
                st.dataframe(sample_df, use_container_width=True, hide_index=True)

    if buscar and customer_id.strip():
        with st.spinner("Buscando cliente…"):
            data = fetch_customer(customer_id)
        if data is None:
            st.warning(f"Cliente `{customer_id}` no encontrado. Probá con otro ID.")
        else:
            st.success(f"✅ Cliente encontrado. RFM pre-cargado.")
            st.session_state["frequency"]      = data["frequency"]
            st.session_state["recency"]        = data["recency"]
            st.session_state["T"]              = data["T"]
            st.session_state["monetary_value"] = data["monetary_value"]

    st.divider()

# ─────────────────────────────────────────────
# SLIDERS RFM (compartidos por ambos modos)
# ─────────────────────────────────────────────
st.markdown("### 🎛️ Valores RFM del Cliente")

col1, col2 = st.columns(2)
with col1:
    frequency = st.slider(
        "**Frequency** — Compras repetidas",
        min_value=0.0, max_value=50.0, step=1.0,
        value=float(st.session_state["frequency"]),
        help="Número de transacciones adicionales (excluyendo la primera compra).",
    )
    recency = st.slider(
        "**Recency** — Días desde 1ª hasta última compra",
        min_value=0.0, max_value=730.0, step=1.0,
        value=float(st.session_state["recency"]),
        help="Cuántos días pasaron entre la primera y la última compra.",
    )

with col2:
    T = st.slider(
        "**T** — Antigüedad del cliente (días)",
        min_value=1.0, max_value=1000.0, step=1.0,
        value=float(st.session_state["T"]),
        help="Días desde la primera compra hasta la fecha de corte de observación.",
    )
    monetary_value = st.slider(
        "**Monetary Value** — Ticket promedio (USD)",
        min_value=0.0, max_value=2000.0, step=5.0,
        value=float(st.session_state["monetary_value"]),
        help="Valor monetario promedio de cada transacción.",
    )

st.divider()

# ─────────────────────────────────────────────
# BOTÓN PREDECIR
# ─────────────────────────────────────────────
col_pred, col_space = st.columns([2, 5])
with col_pred:
    predict_btn = st.button("🚀 Predecir CLV", key="btn_predict", disabled=not api_ok)

if predict_btn:
    if not api_ok:
        st.error("El backend no está disponible. Iniciá el servidor FastAPI primero.")
    else:
        with st.spinner("Calculando CLV…"):
            result = predict_clv(frequency, recency, T, monetary_value, months=12)

        if result:
            st.divider()
            st.markdown("## 📈 Resultados de la Predicción")

            # ── Métricas en cards ──
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="clv-card">
                    <div class="label">CLV 12 Meses</div>
                    <div class="value">${result['clv_predicted']:,.2f}</div>
                    <div class="unit">USD · horizonte 12m</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="clv-card">
                    <div class="label">Compras Esperadas</div>
                    <div class="value">{result['expected_purchases']:.3f}</div>
                    <div class="unit">transacciones</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="clv-card">
                    <div class="label">Ticket Esperado</div>
                    <div class="value">${result['expected_avg_revenue']:,.2f}</div>
                    <div class="unit">USD por compra</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                pct = int(result['prob_alive'] * 100)
                st.markdown(f"""
                <div class="clv-card">
                    <div class="label">Prob. Cliente Activo</div>
                    <div class="value">{pct}%</div>
                    <div class="unit">probabilidad vivo</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge + Horizons ──
            chart_col, horizon_col = st.columns([1, 1])

            with chart_col:
                fig_gauge = make_gauge_chart(result["clv_predicted"])
                st.plotly_chart(fig_gauge, use_container_width=True)

            with horizon_col:
                # Calcular CLV para todos los horizontes seleccionados + 12m
                all_horizons = sorted(set([12] + (extra_horizons or [])))
                horizon_results = []

                with st.spinner("Calculando horizontes adicionales…"):
                    for h in all_horizons:
                        hr = predict_clv(frequency, recency, T, monetary_value, months=h)
                        if hr:
                            horizon_results.append(hr)

                if horizon_results:
                    fig_bar = make_horizon_bar_chart(horizon_results)
                    st.plotly_chart(fig_bar, use_container_width=True)

            # ── Detalle técnico ──
            with st.expander("🔬 Ver detalle técnico completo"):
                st.json(result)
                st.caption(
                    "**Metodología:** El modelo BG/NBD predice el número esperado de transacciones. "
                    "El modelo Gamma-Gamma estima el valor monetario esperado por transacción. "
                    "El CLV se calcula como el producto de ambos."
                )
