# Guía Rápida del Proyecto CLV Forecasting

## 📁 Estructura de Carpetas

### `/configs`
Archivos de configuración en formato YAML para:
- Hiperparámetros de modelos
- Configuración de pipelines de datos
- Parámetros de feature engineering
- Configuración de MLflow y monitoreo

**Archivo principal**: `model_config.yaml`

### `/data`
Gestión de datos con tres subcarpetas:

- **`/raw`**: Datos originales sin procesar (nunca modificar)
- **`/interim`**: Datos intermedios durante el procesamiento
- **`/processed`**: Datos finales listos para modelado

**Nota**: Los archivos de datos están excluidos de Git y deben gestionarse con DVC.

### `/docker`
Configuraciones Docker para diferentes entornos:
- Dockerfile de desarrollo
- Dockerfile de producción
- Configuraciones de Prometheus y Grafana

### `/docs`
Documentación técnica del proyecto:
- Arquitectura del sistema
- Documentación de API (generada automáticamente con FastAPI)
- Guías de uso
- Diagramas de flujo

### `/models`
Modelos serializados y artefactos:
- Archivos `.pkl`, `.joblib` para modelos scikit-learn
- Archivos `.onnx` para modelos optimizados
- Checkpoints de modelos de deep learning
- Metadata de versiones

**Nota**: Los modelos están excluidos de Git y se gestionan con MLflow.

### `/notebooks`
Jupyter notebooks para:
- Análisis exploratorio de datos (EDA)
- Prototipado de modelos
- Experimentación
- Análisis de resultados

**Convención de nombres**: `01_eda.ipynb`, `02_feature_engineering.ipynb`, etc.

### `/src`
Código fuente modular del proyecto:

#### `/src/api`
Implementación de la API REST con FastAPI:
- `main.py`: Punto de entrada de la aplicación
- `endpoints.py`: Definición de endpoints
- `schemas.py`: Modelos Pydantic para validación
- `dependencies.py`: Dependencias inyectables

#### `/src/data_pipeline`
Pipeline de procesamiento de datos:
- `ingestion.py`: Carga de datos desde diversas fuentes
- `cleaning.py`: Limpieza y validación de datos
- `transformation.py`: Transformaciones de datos
- `validation.py`: Validación de calidad de datos

#### `/src/feature_engineering`
Creación de características:
- `rfm_features.py`: Features RFM (Recency, Frequency, Monetary)
- `behavioral_features.py`: Features de comportamiento del cliente
- `temporal_features.py`: Features temporales
- `aggregations.py`: Agregaciones y estadísticas

#### `/src/model_ops`
Operaciones de Machine Learning:
- `train.py`: Script de entrenamiento
- `predict.py`: Script de predicción
- `evaluate.py`: Evaluación de modelos
- `registry.py`: Registro de modelos en MLflow
- `hyperparameter_tuning.py`: Optimización de hiperparámetros

#### `/src/monitoring`
Monitoreo y observabilidad:
- `drift_detection.py`: Detección de drift en datos y modelos
- `metrics.py`: Cálculo de métricas de negocio y técnicas
- `logging_config.py`: Configuración de logging
- `alerts.py`: Sistema de alertas

### `/tests`
Suite de pruebas:

#### `/tests/unit`
Pruebas unitarias de funciones individuales

#### `/tests/integration`
Pruebas de integración de componentes

**Convención**: Archivos de prueba deben comenzar con `test_`

## 🚀 Flujo de Trabajo Típico

### 1. Desarrollo Local
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones
```

### 2. Experimentación
```bash
# Iniciar Jupyter
jupyter notebook

# Trabajar en notebooks/
# Ejemplo: notebooks/01_eda.ipynb
```

### 3. Desarrollo de Features
```bash
# Implementar en src/feature_engineering/
# Escribir pruebas en tests/unit/
pytest tests/unit/test_features.py
```

### 4. Entrenamiento de Modelos
```bash
# Configurar en configs/model_config.yaml
# Entrenar modelo
python src/model_ops/train.py --config configs/model_config.yaml

# Ver resultados en MLflow
mlflow ui
# Abrir http://localhost:5000
```

### 5. Evaluación
```bash
# Evaluar modelo
python src/model_ops/evaluate.py --model-id <model_id>
```

### 6. Despliegue
```bash
# Construir y levantar servicios
docker-compose up --build

# API disponible en http://localhost:8000
# Documentación en http://localhost:8000/docs
```

## 🔧 Comandos Útiles

### Git & DVC
```bash
# Inicializar DVC
dvc init

# Agregar datos a DVC
dvc add data/raw/customers.csv
git add data/raw/customers.csv.dvc .gitignore
git commit -m "Add customer data"

# Pull datos
dvc pull
```

### Testing
```bash
# Todas las pruebas
pytest

# Con cobertura
pytest --cov=src --cov-report=html

# Pruebas específicas
pytest tests/unit/test_features.py::test_rfm_calculation
```

### Docker
```bash
# Construir imagen
docker build -t clv-forecasting .

# Ejecutar contenedor
docker run -p 8000:8000 clv-forecasting

# Ver logs
docker-compose logs -f api
```

### MLflow
```bash
# Iniciar servidor MLflow
mlflow server --host 0.0.0.0 --port 5000

# Registrar modelo
mlflow models serve -m models:/clv_model/Production -p 5001
```

## 📊 Métricas de CLV

### Métricas Principales
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coeficiente de determinación
- **MAPE**: Mean Absolute Percentage Error

### Métricas de Negocio
- **Accuracy de segmentación**: Precisión en clasificar clientes de alto/bajo valor
- **ROI de campañas**: Retorno de inversión basado en predicciones
- **Lift**: Mejora vs. modelo baseline


## 📚 Recursos Adicionales

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 🤝 Contribución

Ver `README.md` para guías de contribución.
