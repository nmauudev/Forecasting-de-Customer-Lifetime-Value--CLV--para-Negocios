# Dockerfile para CLV Forecasting API
FROM python:3.11-slim

# Metadata
LABEL maintainer="CLV Forecasting Team"
LABEL description="Customer Lifetime Value Forecasting API"

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Crear directorios necesarios
RUN mkdir -p logs data/raw data/processed data/interim

# Exponer puerto de la API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
