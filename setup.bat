@echo off
echo ==========================================================
echo  CV Targeting System - Environment Setup Script
echo ==========================================================
echo.

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.9+ and try again.
    pause
    exit /b
)

echo Creating Python virtual environment in '.\venv\'...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b
)

echo Activating virtual environment...
call .\venv\Scripts\activate.bat

echo Installing required Python packages from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please check your network and CUDA/PyTorch setup.
    pause
    exit /b
)

echo.
echo ==========================================================
echo  Setup Complete!
echo ==========================================================
echo.
echo To run the application:
echo 1. Activate the environment: `call venv\Scripts\activate.bat`
echo 2. Run the main script: `python src/main.py`
echo.
pause