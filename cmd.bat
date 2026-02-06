@echo off
echo ========================================
echo PhantomVox Command Environment (System VEnv)
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

echo Virtual environment activated: .venv\
echo.
echo You can now use:
echo   python --version
echo   sox --version
echo   pip list
echo   pip install [package]
echo.
echo To run the app:
echo   python src\main.py
echo.
echo To deactivate, simply close this window.
echo ========================================
echo.

cmd /k
