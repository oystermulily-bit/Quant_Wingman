@echo off
title quant_w1ngman Web Server
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment Python not found:
    echo %PYTHON%
    pause
    exit /b 1
)

echo Starting quant_w1ngman at http://127.0.0.1:8765
echo Keep this window open.
echo.

start "" cmd /c "ping 127.0.0.1 -n 7 >nul 2>&1 & start http://127.0.0.1:8765"

"%PYTHON%" "%~dp0run_web.py" --host 127.0.0.1 --port 8765

echo.
echo Server stopped.
pause
