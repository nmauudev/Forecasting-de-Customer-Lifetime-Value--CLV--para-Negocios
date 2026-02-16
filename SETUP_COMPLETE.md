# 🎉 Configuración del Entorno Completada

## ✅ Resumen de Configuración

### 1. Entorno Virtual Creado
- **Directorio**: `venv/`
- **Python**: 3.13
- **pip**: 26.0.1
- **Estado**: ✅ Activo y funcional

### 2. Dependencias Instaladas (Base)

#### Core Data Science
```
numpy==2.4.2
pandas==2.3.3
scikit-learn==1.8.0
scipy==1.17.0
```

#### CLV Modeling
```
lifetimes==0.11.3  ← Librería especializada para BG/NBD, Pareto/NBD
```

#### MLOps
```
mlflow==3.9.0
```

#### API Framework
```
fastapi==0.129.0
uvicorn==0.40.0
```

#### Monitoring
```
evidently==0.7.20
```

#### Utilities
```
python-dotenv==1.2.1
pyyaml==6.0.3
matplotlib==3.10.8
seaborn==0.13.2
```

**Total de paquetes instalados**: ~130 (incluyendo dependencias)

### 3. Variables de Entorno
- ✅ `.env` creado desde `.env.example`
- ✅ Configuraciones listas para personalizar

### 4. Scripts de Ayuda
- ✅ `activate.bat` - Script de activación rápida

## 🚀 Cómo Empezar

### Activar el Entorno
```cmd
activate.bat
```

O manualmente:
```cmd
venv\Scripts\activate
```

### Verificar Instalación
```cmd
python -c "import lifetimes; print(f'Lifetimes version: {lifetimes.__version__}')"
```

### Iniciar MLflow UI
```cmd
mlflow ui
```
Acceder a: http://localhost:5000

## 📚 Sobre Lifetimes

La librería `lifetimes` es la herramienta estándar para modelado probabilístico de CLV. Incluye:

### Modelos Disponibles
1. **BG/NBD (Beta-Geometric/Negative Binomial Distribution)**
   - Predice frecuencia de compra futura
   - Modela probabilidad de que un cliente esté "vivo"

2. **Pareto/NBD**
   - Alternativa al BG/NBD
   - Mejor para ciertos patrones de comportamiento

3. **Gamma-Gamma**
   - Modela el valor monetario de transacciones
   - Se combina con BG/NBD para CLV completo

### Ejemplo de Uso Básico
```python
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

# Ajustar modelo de frecuencia
bgf = BetaGeoFitter()
bgf.fit(data['frequency'], data['recency'], data['T'])

# Ajustar modelo de valor monetario
ggf = GammaGammaFitter()
ggf.fit(data['frequency'], data['monetary_value'])

# Predecir CLV
clv = ggf.customer_lifetime_value(
    bgf,
    data['frequency'],
    data['recency'],
    data['T'],
    data['monetary_value'],
    time=12  # meses
)
```

## 📁 Estructura del Proyecto

```
Forecasting de Customer Lifetime Value (CLV) para Negocios/
├── venv/                    ← Entorno virtual (NO commitear)
├── .env                     ← Variables de entorno (NO commitear)
├── .env.example             ← Template de variables
├── activate.bat             ← Script de activación
├── requirements.txt         ← Dependencias base
├── configs/                 ← Configuraciones YAML
├── data/                    ← Datos (raw, interim, processed)
├── docs/                    ← Documentación
├── models/                  ← Modelos serializados
├── notebooks/               ← Jupyter notebooks
├── src/                     ← Código fuente
│   ├── api/                 ← FastAPI endpoints
│   ├── data_pipeline/       ← Procesamiento de datos
│   ├── feature_engineering/ ← Creación de features
│   ├── model_ops/           ← Entrenamiento y evaluación
│   └── monitoring/          ← Monitoreo y drift
└── tests/                   ← Pruebas unitarias e integración
```

## 🎯 Próximos Pasos Sugeridos

1. **Agregar datos de ejemplo**
   - Colocar CSV en `data/raw/`
   - Formato esperado: customer_id, purchase_date, amount

2. **Crear notebook de EDA**
   ```cmd
   jupyter notebook
   ```
   - Crear `notebooks/01_exploratory_data_analysis.ipynb`

3. **Implementar pipeline de datos**
   - `src/data_pipeline/ingestion.py`
   - `src/data_pipeline/cleaning.py`

4. **Desarrollar features RFM**
   - `src/feature_engineering/rfm_features.py`
   - Calcular Recency, Frequency, Monetary

5. **Entrenar modelo CLV**
   - `src/model_ops/train.py`
   - Usar lifetimes BG/NBD + Gamma-Gamma

6. **Crear API de predicción**
   - `src/api/main.py`
   - Endpoints para predicciones en tiempo real

## 📖 Recursos Útiles

### Documentación
- [Lifetimes Documentation](https://lifetimes.readthedocs.io/)
- [Lifetimes GitHub](https://github.com/CamDavidsonPilon/lifetimes)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Tutoriales de CLV con Lifetimes
- [Customer Lifetime Value with Python](https://towardsdatascience.com/customer-lifetime-value-with-python-3c5e8c4e8d2f)
- [Probabilistic CLV Modeling](https://www.kaggle.com/code/janiobachmann/customer-lifetime-value-clv-prediction)

### Papers de Referencia
- Fader & Hardie (2005): "A Note on Deriving the Pareto/NBD Model"
- Fader et al. (2005): "RFM and CLV: Using Iso-value Curves for Customer Base Analysis"

## ⚙️ Configuración Adicional (Opcional)

### Instalar Jupyter
```cmd
pip install jupyter ipykernel
python -m ipykernel install --user --name=clv-env
```

### Instalar herramientas de desarrollo
```cmd
pip install pytest black flake8 isort
```

### Configurar DVC (Data Version Control)
```cmd
pip install dvc dvc-s3
dvc init
```

## 🎊 ¡Todo Listo!

El entorno está completamente configurado y listo para comenzar el desarrollo del proyecto de forecasting de CLV.

**Comando para empezar**:
```cmd
activate.bat
```

---

**Fecha de configuración**: 2026-02-16
**Estado**: ✅ Completado exitosamente
