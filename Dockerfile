# ─────────────────────────────────────────────
# Dockerfile — Backend FastAPI (clv_api)
# Puerto: 8000
# ─────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="CLV Forecasting Team"
LABEL description="CLV Forecasting — FastAPI Backend"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias de sistema mínimas (curl para el healthcheck, gcc para lifetimes/autograd)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar solo las dependencias del backend
COPY requirements.api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.api.txt

# Copiar código fuente y artefactos del modelo
COPY src/     ./src/
COPY models/  ./models/
COPY data/processed/ ./data/processed/
COPY configs/ ./configs/

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
