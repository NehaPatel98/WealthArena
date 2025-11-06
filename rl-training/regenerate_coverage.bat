@echo off
REM Regenerate coverage.xml with correct paths for SonarQube

echo ========================================
echo   Regenerating Coverage for SonarQube
echo ========================================
echo.

REM Step 1: Check if we're in the right directory
if not exist "pytest.ini" (
    echo ERROR: Please run this script from the wealtharena_rl directory!
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [1/4] Current directory: %CD%
echo.

REM Step 2: Delete old coverage file
echo [2/4] Deleting old coverage.xml...
if exist "coverage.xml" (
    del /f coverage.xml
    echo      Old coverage.xml deleted.
) else (
    echo      No old coverage.xml found (OK).
)
echo.

REM Step 3: Install/upgrade dependencies
echo [3/4] Installing dependencies...
pip install pytest pytest-cov --quiet
echo      Dependencies ready.
echo.

REM Step 4: Generate new coverage
echo [4/4] Generating coverage report...
echo ----------------------------------------
pytest --cov=. --cov-report=xml --cov-report=term
echo ----------------------------------------
echo.

REM Verify coverage.xml was created
if exist "coverage.xml" (
    echo.
    echo ========================================
    echo   SUCCESS! Coverage Generated
    echo ========================================
    echo.
    echo Coverage file: %CD%\coverage.xml
    echo.
    echo Next steps:
    echo 1. Verify paths in coverage.xml look correct
    echo 2. Run SonarQube scan
    echo 3. Check SonarQube dashboard for coverage percentage
    echo.
    echo To verify paths, run:
    echo   findstr /C:"filename=" coverage.xml
    echo.
) else (
    echo.
    echo ========================================
    echo   ERROR: Coverage Generation Failed
    echo ========================================
    echo.
    echo coverage.xml was not created.
    echo.
    echo Possible issues:
    echo - Tests may have failed
    echo - pytest-cov not installed correctly
    echo - Configuration issue in pytest.ini or .coveragerc
    echo.
    echo Check the output above for error messages.
    echo.
)

pause

