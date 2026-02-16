@echo off
REM Script de activacion del entorno virtual para CLV Forecasting
REM Uso: activate.bat

echo ========================================
echo   CLV Forecasting - Entorno Virtual
echo ========================================
echo.

REM Activar el entorno virtual
call venv\Scripts\activate.bat

echo.
echo Entorno virtual activado!
echo.
echo Dependencias instaladas:
echo   - pandas, numpy, scikit-learn
echo   - lifetimes (CLV modeling)
echo   - mlflow (experiment tracking)
echo   - fastapi, uvicorn (API)
echo   - evidently (monitoring)
echo.
echo Para desactivar: deactivate
echo ========================================
