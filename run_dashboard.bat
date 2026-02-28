@echo off
title Live Quant Dashboard: PyTorch Deep BSDE vs Institutional Baselines
color 0a

echo ============================================================
echo   Starting the Real-Time Streamlit Quant Engine...
echo ============================================================
echo.

:: Launch Streamlit via the active virtual environment executing the live dashboard
venv\Scripts\python.exe -m streamlit run src\app.py

echo.
pause
