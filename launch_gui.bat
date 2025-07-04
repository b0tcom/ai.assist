@echo off
echo Starting AI Aim Assist with GUI...
cd /d "%~dp0"
.venv\Scripts\python.exe thebot\src\main.py --ui gui
pause
