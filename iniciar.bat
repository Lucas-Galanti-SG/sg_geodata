@echo off
chcp 65001 >nul
cd /d "%~dp0"

python _check_deps.py
if %errorlevel% neq 0 (
    echo.
    echo  Abortado. Corrija as dependencias e tente novamente.
    pause
    exit /b 1
)

echo  Iniciando SGGeoData...
echo.
streamlit run app.py
pause
