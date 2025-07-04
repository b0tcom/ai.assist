@echo off
echo Starting AI Aim Assist in headless mode...
cd /d "%~dp0"
.venv\Scripts\python.exe thebot\src\main.py --ui headless
pause
