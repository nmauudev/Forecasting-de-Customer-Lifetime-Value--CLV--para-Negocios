# Verificación del Entorno - CLV Forecasting

## ✅ Estado del Entorno Virtual

### Entorno Creado
- **Ubicación**: `venv/`
- **Python**: Python 3.13
- **Gestor de paquetes**: pip 26.0.1

### Variables de Entorno
- **Archivo**: `.env` (creado desde `.env.example`)
- **Estado**: Configurado ✓

## 📦 Dependencias Instaladas

### Core Data Science
- ✅ **numpy** 2.4.2
- ✅ **pandas** 2.3.3
- ✅ **scikit-learn** 1.8.0
- ✅ **scipy** 1.17.0

### CLV Modeling
- ✅ **lifetimes** 0.11.3
  - Modelos: BG/NBD, Pareto/NBD, Gamma-Gamma
  - Para predicción de CLV probabilístico

### MLOps & Experiment Tracking
- ✅ **mlflow** 3.9.0
  - Tracking de experimentos
  - Registro de modelos
  - Versionado de artefactos

### API Framework
- ✅ **fastapi** 0.129.0
- ✅ **uvicorn** 0.40.0
- ✅ **pydantic** 2.12.5
- ✅ **starlette** 0.52.1

### Monitoring & Drift Detection
- ✅ **evidently** 0.7.20
  - Detección de drift
  - Monitoreo de calidad de datos
  - Métricas de modelo

### Utilities
- ✅ **python-dotenv** 1.2.1
- ✅ **pyyaml** 6.0.3

### Visualization
- ✅ **matplotlib** 3.10.8
- ✅ **seaborn** 0.13.2
- ✅ **plotly** 5.24.1

## 🚀 Cómo Activar el Entorno

### Opción 1: Script de Activación (Recomendado)
```cmd
activate.bat
```

### Opción 2: Activación Manual
```cmd
venv\Scripts\activate
```

## 🧪 Verificar Instalación

Para verificar que todas las dependencias están correctamente instaladas:

```cmd
venv\Scripts\python.exe -c "import pandas; import numpy; import sklearn; import lifetimes; import mlflow; import fastapi; import evidently; print('OK: Todo instalado correctamente!')"
```

## 📝 Próximos Pasos

1. **Activar el entorno virtual**
   ```cmd
   activate.bat
   ```

2. **Verificar MLflow**
   ```cmd
   mlflow --version
   ```

3. **Iniciar servidor MLflow** (opcional)
   ```cmd
   mlflow ui
   ```
   Acceder a: http://localhost:5000

4. **Probar FastAPI** (opcional)
   ```cmd
   uvicorn --version
   ```

5. **Comenzar desarrollo**
   - Agregar datos en `data/raw/`
   - Crear notebooks en `notebooks/`
   - Desarrollar código en `src/`

## 🔧 Comandos Útiles

### Gestión de Paquetes
```cmd
# Listar paquetes instalados
pip list

# Instalar paquete adicional
pip install nombre-paquete

# Actualizar requirements.txt
pip freeze > requirements.txt

# Instalar desde requirements.txt
pip install -r requirements.txt
```

### Desarrollo
```cmd
# Ejecutar script Python
python src/script.py

# Ejecutar con módulo
python -m src.module_name

# Ejecutar tests (cuando se implementen)
pytest tests/
```

### API
```cmd
# Iniciar servidor de desarrollo
uvicorn src.api.main:app --reload

# Iniciar en puerto específico
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### MLflow
```cmd
# Iniciar UI de MLflow
mlflow ui

# Iniciar en puerto específico
mlflow ui --port 5000

# Ver experimentos
mlflow experiments list
```

## ⚠️ Notas Importantes

1. **Siempre activar el entorno** antes de trabajar en el proyecto
2. **No commitear** el directorio `venv/` a Git (ya está en `.gitignore`)
3. **Actualizar** `requirements.txt` cuando agregues nuevas dependencias
4. **Usar** `.env` para variables de entorno sensibles (no commitear)

## 🐛 Troubleshooting

### Problema: No se puede activar el entorno
**Solución**: Verificar que el entorno fue creado correctamente
```cmd
python -m venv venv
```

### Problema: Módulo no encontrado
**Solución**: Verificar que el entorno está activado y el paquete instalado
```cmd
pip list | findstr nombre-paquete
```

### Problema: Error de importación
**Solución**: Reinstalar dependencias
```cmd
pip install -r requirements.txt --force-reinstall
```

## 📚 Recursos

- [Documentación de lifetimes](https://lifetimes.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Evidently Documentation](https://docs.evidentlyai.com/)

---

**Última actualización**: 2026-02-16
**Estado**: ✅ Entorno configurado y listo para desarrollo
