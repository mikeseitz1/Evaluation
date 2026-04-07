# Telco Customer Churn Project Startup Script (PowerShell)
# This script starts all project services: FastAPI, Streamlit, and MLflow UI

param(
    [switch]$NoMLflow = $false,
    [switch]$NoFastAPI = $false,
    [switch]$NoStreamlit = $false
)

# Get the script directory
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

# Check if virtual environment exists
$VenvPath = Join-Path $ProjectDir ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "Error: Virtual environment not found at $VenvPath" -ForegroundColor Red
    Write-Host "Please create a virtual environment first with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Activating Python Virtual Environment..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

& "$VenvPath\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Could not activate virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Telco Churn Prediction Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ServicesStarted = @()

# Start FastAPI Server
if (-not $NoFastAPI) {
    Write-Host "Starting FastAPI Server (port 8000)..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
    $ServicesStarted += "FastAPI Server"
    Start-Sleep -Seconds 2
}

# Start Streamlit App
if (-not $NoStreamlit) {
    Write-Host "Starting Streamlit App (port 8501)..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "streamlit run streamlit_app.py"
    $ServicesStarted += "Streamlit App"
    Start-Sleep -Seconds 2
}

# Start MLflow UI
if (-not $NoMLflow) {
    Write-Host "Starting MLflow UI (port 5000)..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "mlflow ui --backend-store-uri sqlite:///mlflow.db"
    $ServicesStarted += "MLflow UI"
}

# Display summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Services Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Started $($ServicesStarted.Count) service(s):" -ForegroundColor Green
foreach ($Service in $ServicesStarted) {
    Write-Host "  ✓ $Service" -ForegroundColor Green
}

Write-Host ""
Write-Host "Access points:" -ForegroundColor Yellow
Write-Host "  FastAPI Server:  http://localhost:8000" -ForegroundColor White
Write-Host "  FastAPI Docs:    http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Streamlit App:   http://localhost:8501" -ForegroundColor White
if ($ServicesStarted -contains "MLflow UI") {
    Write-Host "  MLflow UI:       http://localhost:5000" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Tip: Check the opened windows to see service output" -ForegroundColor Yellow
Write-Host "Tip: Close any window to stop that service" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($ServicesStarted.Count -gt 0) {
    Write-Host "All services are running!" -ForegroundColor Green
} else {
    Write-Host "Warning: No services were started" -ForegroundColor Red
}

Write-Host ""
