@echo off
REM WealthArena Demo Startup Script for Windows
REM This script starts all services needed for the demo

echo 🚀 Starting WealthArena Demo...
echo.

REM Set project root
cd /d "%~dp0\.."
set PROJECT_ROOT=%CD%
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

echo 📁 Creating directories...
if not exist "%PROJECT_ROOT%\data\raw" mkdir "%PROJECT_ROOT%\data\raw"
if not exist "%PROJECT_ROOT%\data\processed" mkdir "%PROJECT_ROOT%\data\processed"
if not exist "%PROJECT_ROOT%\data\features" mkdir "%PROJECT_ROOT%\data\features"
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"
if not exist "%PROJECT_ROOT%\airflow\dags" mkdir "%PROJECT_ROOT%\airflow\dags"

REM Check if virtual environment exists
if not exist "%PROJECT_ROOT%\venv" (
    echo ⚠️  Virtual environment not found. Creating...
    python -m venv "%PROJECT_ROOT%\venv"
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call "%PROJECT_ROOT%\venv\Scripts\activate.bat"

REM Install dependencies
echo 📦 Installing dependencies...
pip install -q -r "%PROJECT_ROOT%\requirements_demo.txt"

REM Initialize Airflow
set AIRFLOW_HOME=%PROJECT_ROOT%\airflow
if not exist "%AIRFLOW_HOME%\airflow.db" (
    echo 🔄 Initializing Airflow...
    airflow db init
    
    REM Create admin user
    airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@wealtharena.com --password admin
)

echo.
echo ✅ Setup complete!
echo.
echo Starting services...
echo.

REM Start Airflow Webserver
echo 🌐 Starting Airflow Webserver on http://localhost:8080
start /B airflow webserver --port 8080 > "%PROJECT_ROOT%\logs\airflow_webserver.log" 2>&1

REM Wait a bit
timeout /t 3 /nobreak > nul

REM Start Airflow Scheduler
echo 📅 Starting Airflow Scheduler
start /B airflow scheduler > "%PROJECT_ROOT%\logs\airflow_scheduler.log" 2>&1

REM Wait
timeout /t 2 /nobreak > nul

REM Start Backend API
echo 🔌 Starting Backend API on http://localhost:8000
start /B python backend\main.py > "%PROJECT_ROOT%\logs\backend.log" 2>&1

REM Wait
timeout /t 2 /nobreak > nul

echo.
echo ============================================
echo 🎉 WealthArena Demo is Running!
echo ============================================
echo.
echo 📊 Services:
echo   • Airflow UI:    http://localhost:8080 (admin/admin)
echo   • Backend API:   http://localhost:8000
echo   • API Docs:      http://localhost:8000/docs
echo.
echo 📝 Logs:
echo   • Airflow Web:   tail -f logs\airflow_webserver.log
echo   • Airflow Sched: tail -f logs\airflow_scheduler.log
echo   • Backend:       tail -f logs\backend.log
echo.
echo 🎮 Next Steps:
echo   1. Open Airflow UI and trigger 'fetch_market_data' DAG
echo   2. After data fetch, trigger 'preprocess_market_data' DAG
echo   3. Check API at http://localhost:8000/docs
echo   4. Start the frontend (see DEMO_SETUP_GUIDE.md)
echo.
echo Press Ctrl+C to stop
echo.

pause

