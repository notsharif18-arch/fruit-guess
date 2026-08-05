@echo off
title FruitVision AI Setup

echo ============================================
echo        FruitVision AI Installer
echo ============================================
echo.

:: Check Python
py --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 is not installed.
    pause
    exit
)

echo Creating virtual environment...
py -m venv venv

echo.
echo Activating environment...
call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Downloading dataset...
python download_dataset.py

echo.
echo ============================================
echo Installation Complete!
echo Double-click run.bat to launch the project.
echo ============================================

pause
