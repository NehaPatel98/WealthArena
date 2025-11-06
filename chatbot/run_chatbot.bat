@echo off
REM WealthArena - Chatbot Service Startup
REM This script starts the chatbot service on port 5001

echo ========================================
echo WealthArena - Chatbot Service
echo ========================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Set Python path (adjust if needed)
set PYTHON_EXE=python

REM Check if Python is available
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found
    if exist ".env.example" (
        echo Copying .env.example to .env...
        copy .env.example .env >nul
        echo .env file created from .env.example
        echo.
        echo IMPORTANT: Please review .env and set your GROQ_API_KEY
        echo.
        pause
        exit /b 1
    ) else (
        echo ERROR: .env.example file not found
        echo Cannot create .env file. Exiting.
        pause
        exit /b 1
    )
)

REM Create data directories if they don't exist
if not exist "data\chroma_db" mkdir data\chroma_db
if not exist "data\chat_history" mkdir data\chat_history

REM Display startup info
echo Starting chatbot service...
echo Port: 5001
echo.
echo Press CTRL+C to stop the service
echo.

REM Run the service
%PYTHON_EXE% main.py

REM If service exits
echo.
echo ========================================
echo Chatbot service stopped
echo ========================================
pause

