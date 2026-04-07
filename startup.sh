#!/bin/bash
# Telco Customer Churn Project Startup Script
# This script starts all project services: FastAPI, Streamlit, and MLflow UI

set -e

# Get the script directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Parse arguments
START_MLFLOW=true
START_FASTAPI=true
START_STREAMLIT=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-mlflow)
            START_MLFLOW=false
            shift
            ;;
        --no-fastapi)
            START_FASTAPI=false
            shift
            ;;
        --no-streamlit)
            START_STREAMLIT=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-mlflow      Do not start MLflow UI"
            echo "  --no-fastapi     Do not start FastAPI server"
            echo "  --no-streamlit   Do not start Streamlit app"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please create a virtual environment first with: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo ""
echo "========================================"
echo "Activating Python Virtual Environment..."
echo "========================================"
source .venv/bin/activate

echo ""
echo "========================================"
echo "Starting Telco Churn Prediction Services"
echo "========================================"
echo ""

SERVICES_STARTED=()

# Start FastAPI
if [ "$START_FASTAPI" = true ]; then
    echo "Starting FastAPI Server (port 8000)..."
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload > /tmp/fastapi.log 2>&1 &
    FASTAPI_PID=$!
    echo "✓ FastAPI Server started (PID: $FASTAPI_PID)"
    SERVICES_STARTED+=("FastAPI Server (PID: $FASTAPI_PID)")
    sleep 2
fi

# Start Streamlit
if [ "$START_STREAMLIT" = true ]; then
    echo "Starting Streamlit App (port 8501)..."
    streamlit run streamlit_app.py > /tmp/streamlit.log 2>&1 &
    STREAMLIT_PID=$!
    echo "✓ Streamlit App started (PID: $STREAMLIT_PID)"
    SERVICES_STARTED+=("Streamlit App (PID: $STREAMLIT_PID)")
    sleep 2
fi

# Start MLflow UI
if [ "$START_MLFLOW" = true ]; then
    echo "Starting MLflow UI (port 5000)..."
    mlflow ui --backend-store-uri sqlite:///mlflow.db > /tmp/mlflow.log 2>&1 &
    MLFLOW_PID=$!
    echo "✓ MLflow UI started (PID: $MLFLOW_PID)"
    SERVICES_STARTED+=("MLflow UI (PID: $MLFLOW_PID)")
fi

# Display summary
echo ""
echo "========================================"
echo "Services Status"
echo "========================================"
echo ""
echo "Started ${#SERVICES_STARTED[@]} service(s):"
for service in "${SERVICES_STARTED[@]}"; do
    echo "  ✓ $service"
done

echo ""
echo "Access points:"
echo "  FastAPI Server:  http://localhost:8000"
echo "  FastAPI Docs:    http://localhost:8000/docs"
echo "  Streamlit App:   http://localhost:8501"
if [ "$START_MLFLOW" = true ]; then
    echo "  MLflow UI:       http://localhost:5000"
fi

echo ""
echo "========================================"
echo "Tip: Check logs in /tmp/*.log"
echo "Tip: To stop services, run:"
for service in "${SERVICES_STARTED[@]}"; do
    PID=$(echo $service | grep -oP 'PID: \K[0-9]+')
    if [ ! -z "$PID" ]; then
        echo "  kill $PID"
    fi
done
echo "========================================"
echo ""

if [ ${#SERVICES_STARTED[@]} -gt 0 ]; then
    echo "All services are running!"
else
    echo "Warning: No services were started"
    exit 1
fi

echo ""
