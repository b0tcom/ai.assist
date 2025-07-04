@echo off
echo Starting AI Aim Assist as Administrator...
cd /d "%~dp0"
powershell -Command "Start-Process cmd -ArgumentList '/c .venv\Scripts\python.exe thebot\src\main.py --ui gui & pause' -Verb RunAs"
