@echo off
REM Telco Customer Churn Project Startup Script
REM This script starts all project services: FastAPI, Streamlit, and MLflow UI

setlocal enabledelayedexpansion

REM Get the project directory
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

REM Activate virtual environment
echo.
echo ========================================
echo Activating Python Virtual Environment...
echo ========================================
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo Error: Could not activate virtual environment
    echo Make sure .venv directory exists and is properly set up
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Telco Churn Prediction Services
echo ========================================
echo.

REM Start FastAPI server in a new window
echo Starting FastAPI Server (port 8000)...
start cmd /k "title FastAPI Server && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for FastAPI to start
timeout /t 2 /nobreak

REM Start Streamlit app in a new window
echo Starting Streamlit App (port 8501)...
start cmd /k "title Streamlit App && streamlit run streamlit_app.py"

REM Wait a moment
timeout /t 2 /nobreak

REM Ask user if they want to start MLflow UI
echo.
set /p START_MLFLOW="Start MLflow UI as well? (y/n): "
if /i "%START_MLFLOW%"=="y" (
    echo Starting MLflow UI (port 5000)...
    start cmd /k "title MLflow UI && mlflow ui --backend-store-uri sqlite:///mlflow.db"
)

echo.
echo ========================================
echo Services Starting...
echo ========================================
echo.
echo FastAPI Server:  http://localhost:8000
echo FastAPI Docs:    http://localhost:8000/docs
echo Streamlit App:   http://localhost:8501
if /i "%START_MLFLOW%"=="y" (
    echo MLflow UI:      http://localhost:5000
)
echo.
echo Press any of the service windows to interact with them.
echo Close any window to stop that service.
echo.
echo To stop all services, close their respective command windows.
echo ========================================
echo.

pause
