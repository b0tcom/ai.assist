@echo off
setlocal enabledelayedexpansion

REM =========================================================================
REM AI Aim Assist System - Startup Script
REM =========================================================================
REM Purpose: Launch the AI aim assist system with proper initialization
REM Author: AI Gaming Analysis System
REM Version: 2.0.0
REM =========================================================================

echo.
echo ==========================================
echo   AI Aim Assist System - Starting...
echo ==========================================
echo.

REM Set colors for better visibility
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

REM Get the directory where this batch file is located
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo %BLUE%Project Directory: %PROJECT_ROOT%%RESET%
echo.

REM =========================================================================
REM 1. ENVIRONMENT CHECKS
REM =========================================================================

echo %YELLOW%[1/6] Checking Python Environment...%RESET%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python not found in PATH%RESET%
    echo Please install Python 3.8+ or activate your virtual environment
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%✓ Python %PYTHON_VERSION% found%RESET%

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo %GREEN%✓ Virtual environment found%RESET%
    call .venv\Scripts\activate.bat
    echo %GREEN%✓ Virtual environment activated%RESET%
) else (
    echo %YELLOW%⚠ No virtual environment found, using system Python%RESET%
)

REM =========================================================================
REM 2. DEPENDENCY CHECKS
REM =========================================================================

echo.
echo %YELLOW%[2/6] Checking Dependencies...%RESET%

REM Check for requirements.txt and install if needed
if exist "requirements.txt" (
    echo %BLUE%Checking Python packages...%RESET%
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo %RED%ERROR: Failed to install dependencies%RESET%
        echo Run: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo %GREEN%✓ Dependencies verified%RESET%
) else (
    echo %YELLOW%⚠ No requirements.txt found, proceeding anyway%RESET%
)

REM =========================================================================
REM 3. CONFIGURATION CHECKS
REM =========================================================================

echo.
echo %YELLOW%[3/6] Checking Configuration...%RESET%

REM Check for config files
if exist "configs\config.ini" (
    echo %GREEN%✓ Main config file found%RESET%
) else (
    echo %YELLOW%⚠ Main config not found, will use defaults%RESET%
    REM Create configs directory if it doesn't exist
    if not exist "configs" mkdir configs
)

if exist "configs\default_config.json" (
    echo %GREEN%✓ Default config found%RESET%
) else (
    echo %YELLOW%⚠ Default config not found%RESET%
)

REM =========================================================================
REM 4. DIRECTORY STRUCTURE CHECKS
REM =========================================================================

echo.
echo %YELLOW%[4/6] Checking Directory Structure...%RESET%

REM Check for main source directory
if exist "thebot\src\main.py" (
    echo %GREEN%✓ Main source files found%RESET%
) else (
    echo %RED%ERROR: Main source files not found%RESET%
    echo Expected: thebot\src\main.py
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "logs" (
    mkdir logs
    echo %GREEN%✓ Created logs directory%RESET%
) else (
    echo %GREEN%✓ Logs directory exists%RESET%
)

if not exist "profiles" (
    mkdir profiles
    echo %GREEN%✓ Created profiles directory%RESET%
) else (
    echo %GREEN%✓ Profiles directory exists%RESET%
)

REM =========================================================================
REM 5. HARDWARE CHECKS
REM =========================================================================

echo.
echo %YELLOW%[5/6] Checking Hardware...%RESET%

REM Check for Arduino (optional)
echo %BLUE%Scanning for Arduino devices...%RESET%
python -c "import serial.tools.list_ports; ports = list(serial.tools.list_ports.comports()); arduino_ports = [p for p in ports if 'arduino' in p.description.lower()]; print('Arduino found on:', arduino_ports[0].device if arduino_ports else 'None')" 2>nul
if errorlevel 1 (
    echo %YELLOW%⚠ Could not scan for Arduino (serial library issue)%RESET%
) else (
    echo %GREEN%✓ Hardware scan completed%RESET%
)

REM =========================================================================
REM 6. LAUNCH APPLICATION
REM =========================================================================

echo.
echo %YELLOW%[6/6] Launching Application...%RESET%
echo.

REM Parse command line arguments for startup options
set "CONFIG_FILE=configs\config.ini"
set "UI_MODE="
set "DEBUG_MODE="

REM Check for command line arguments
if "%1"=="--gui" set "UI_MODE=--ui gui"
if "%1"=="--overlay" set "UI_MODE=--ui overlay"
if "%1"=="--headless" set "UI_MODE=--ui headless"
if "%1"=="--debug" set "DEBUG_MODE=--debug"
if "%2"=="--debug" set "DEBUG_MODE=--debug"

REM Display startup options
echo %BLUE%Startup Configuration:%RESET%
echo   Config File: %CONFIG_FILE%
if defined UI_MODE echo   UI Mode: %UI_MODE%
if defined DEBUG_MODE echo   Debug Mode: Enabled
echo   Working Directory: %CD%
echo.

REM Launch the application
echo %GREEN%Starting AI Aim Assist System...%RESET%
echo %BLUE%===========================================%RESET%
echo.

REM Use python -m to run as module for proper imports
python -m thebot.src.main --config "%CONFIG_FILE%" %UI_MODE% %DEBUG_MODE%

REM Capture exit code
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo %BLUE%===========================================%RESET%

REM Handle exit codes
if %EXIT_CODE% equ 0 (
    echo %GREEN%✓ Application exited normally%RESET%
) else (
    echo %RED%✗ Application exited with error code: %EXIT_CODE%%RESET%
    echo.
    echo %YELLOW%Troubleshooting Tips:%RESET%
    echo - Check the logs directory for error details
    echo - Ensure Arduino is connected (if using hardware mode)
    echo - Verify configuration files are correct
    echo - Try running with --debug flag for more information
    echo.
    echo %BLUE%To run with debug mode:%RESET%
    echo   start.bat --debug
    echo.
    echo %BLUE%To run with specific UI:%RESET%
    echo   start.bat --gui
    echo   start.bat --overlay  
    echo   start.bat --headless
    echo.
)

REM Keep window open if there was an error
if %EXIT_CODE% neq 0 (
    echo Press any key to exit...
    pause >nul
)

endlocal
exit /b %EXIT_CODE%