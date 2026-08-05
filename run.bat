@echo off
title FruitVision AI

if not exist venv (
    echo.
    echo ==========================================
    echo  Project is not set up yet.
    echo  Please run setup.bat first.
    echo ==========================================
    pause
    exit
)

call venv\Scripts\activate

echo.
echo Starting FruitVision AI...
echo.

streamlit run app.py

pause
