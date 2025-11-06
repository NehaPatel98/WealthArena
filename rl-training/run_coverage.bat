@echo off
REM Batch script to run pytest with coverage for SonarQube
REM Usage: run_coverage.bat

echo ============================================================
echo Running Tests with Coverage for SonarQube
echo ============================================================
echo.

REM Check if pytest and pytest-cov are installed
python -c "import pytest, pytest_cov" 2>NUL
if errorlevel 1 (
    echo Installing pytest and pytest-cov...
    pip install pytest pytest-cov
    echo.
)

echo Running tests with coverage...
echo.

REM Run pytest with coverage
pytest --cov=. --cov-report=xml --cov-report=term -v

echo.
echo ============================================================

if exist coverage.xml (
    echo [SUCCESS] coverage.xml generated successfully!
    echo.
    echo File location: %cd%\coverage.xml
    echo.
    echo Next steps:
    echo 1. Upload coverage.xml to SonarQube
    echo 2. Commit your code (but NOT coverage.xml - already in .gitignore)
) else (
    echo [WARNING] coverage.xml not found
)

echo ============================================================
echo.

pause

