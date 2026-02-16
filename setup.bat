@echo off
REM MedClinic Quick Start Setup Script for Windows
REM Run: setup.bat

echo.
echo 7 MedClinic Setup
echo ====================
echo.

REM Check Python version
echo Checking Python...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+ from python.org
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To start the server:
echo   venv\Scripts\activate.bat
echo   python main.py
echo.
echo Then open: http://localhost:8000
echo.
pause
