@echo off
echo ========================================
echo Starting PhantomVox (System VEnv)
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_env_venv.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call "%~dp0.venv\Scripts\activate.bat"

python src\main.py

echo.
echo ========================================
echo Program exited
echo ========================================
pause
