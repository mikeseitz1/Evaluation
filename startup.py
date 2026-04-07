#!/usr/bin/env python3
"""
Telco Customer Churn Project Startup Script

This script starts all project services:
- FastAPI Server (port 8000)
- Streamlit App (port 8501)
- MLflow UI (port 5000) - optional

Usage:
    python startup.py
    python startup.py --no-mlflow
    python startup.py --help
"""

import subprocess
import sys
import time
import os
import argparse
from pathlib import Path


def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":
        venv_python = Path(".venv/Scripts/python.exe")
    else:
        venv_python = Path(".venv/bin/python")
    
    if venv_python.exists():
        return str(venv_python)
    else:
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please create a virtual environment first with: python -m venv .venv")
        sys.exit(1)


def activate_venv():
    """Activate the virtual environment for subprocess calls."""
    if sys.platform == "win32":
        activate_script = Path(".venv/Scripts/activate.bat")
    else:
        activate_script = Path(".venv/bin/activate")
    
    if not activate_script.exists():
        print(f"Error: Virtual environment activation script not found")
        sys.exit(1)


def start_service(name, command, shell=False):
    """Start a service in a new process."""
    print(f"Starting {name}...")
    try:
        # On Windows, use 'start' command; on Unix, use subprocess.Popen
        if sys.platform == "win32":
            subprocess.Popen(
                command,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                shell=True
            )
        else:
            subprocess.Popen(
                command,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        print(f"✓ {name} started")
        return True
    except Exception as e:
        print(f"✗ Failed to start {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Start all Telco Churn Prediction Project services"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Do not start MLflow UI"
    )
    parser.add_argument(
        "--no-fastapi",
        action="store_true",
        help="Do not start FastAPI server"
    )
    parser.add_argument(
        "--no-streamlit",
        action="store_true",
        help="Do not start Streamlit app"
    )
    
    args = parser.parse_args()
    
    # Get the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("\n" + "="*50)
    print("Telco Customer Churn Prediction Project")
    print("Starting Services...")
    print("="*50 + "\n")
    
    # Verify virtual environment exists
    venv_python = get_venv_python()
    activate_venv()
    
    services_started = []
    
    # Start FastAPI
    if not args.no_fastapi:
        if sys.platform == "win32":
            cmd = f'start "FastAPI Server" cmd /k "{venv_python} -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"'
        else:
            cmd = f'source .venv/bin/activate && python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload &'
        
        if start_service("FastAPI Server (port 8000)", cmd):
            services_started.append("FastAPI Server")
        time.sleep(2)
    
    # Start Streamlit
    if not args.no_streamlit:
        if sys.platform == "win32":
            cmd = f'start "Streamlit App" cmd /k "{venv_python} -m streamlit run streamlit_app.py"'
        else:
            cmd = f'source .venv/bin/activate && streamlit run streamlit_app.py &'
        
        if start_service("Streamlit App (port 8501)", cmd):
            services_started.append("Streamlit App")
        time.sleep(2)
    
    # Start MLflow UI
    if not args.no_mlflow:
        if sys.platform == "win32":
            cmd = f'start "MLflow UI" cmd /k "{venv_python} -m mlflow ui --backend-store-uri sqlite:///mlflow.db"'
        else:
            cmd = f'source .venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db &'
        
        if start_service("MLflow UI (port 5000)", cmd):
            services_started.append("MLflow UI")
    
    # Display summary
    print("\n" + "="*50)
    print("Services Status")
    print("="*50)
    print(f"\nStarted {len(services_started)} service(s):")
    for service in services_started:
        print(f"  ✓ {service}")
    
    print("\nAccess points:")
    print("  FastAPI Server:  http://localhost:8000")
    print("  FastAPI Docs:    http://localhost:8000/docs")
    print("  Streamlit App:   http://localhost:8501")
    if "MLflow UI" in services_started:
        print("  MLflow UI:       http://localhost:5000")
    
    print("\n" + "="*50)
    print("Tip: Check the opened windows to see service output")
    print("Tip: Close any window to stop that service")
    print("="*50 + "\n")
    
    if services_started:
        print("All services are running!")
    else:
        print("Warning: No services were started")
        sys.exit(1)


if __name__ == "__main__":
    main()
