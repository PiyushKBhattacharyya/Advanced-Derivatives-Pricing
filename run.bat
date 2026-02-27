@echo off
title Deep BSDE Pipeline Orchestrator
color 0f

echo ============================================================
echo   Starting Quantitative Pricing Pipeline Execution...
echo ============================================================
echo.

:: Execute the central orchestrator via the local Windows venv
venv\Scripts\python.exe run_pipeline.py

echo.
pause
