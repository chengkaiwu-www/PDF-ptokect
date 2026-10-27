@echo off
chcp 65001 >nul
cd /d "%~dp0"

set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set PYTHON=D:\anaconda\python.exe

echo ============================================
echo  RAG local web server
echo  open http://127.0.0.1:8000 after "ready"
echo  press Ctrl+C to stop
echo ============================================
echo.

"%PYTHON%" -u server\app.py

echo.
echo [server exited] press any key to close...
pause >nul
