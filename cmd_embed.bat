@echo off
echo ========================================
echo  PhantomVox Command Environment (Embed)
echo ========================================
echo.

REM Check if embedded Python exists
if not exist "%~dp0python-3.12.9-embed\python.exe" (
    echo ERROR: Embedded Python not found!
    echo Please ensure python-3.12.9-embed folder exists.
    pause
    exit /b 1
)

REM Set project root directory
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM Add src directory to Python search path
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

REM Add env folder to PATH for sox, wget and other tools
set "PATH=%PROJECT_ROOT%\env;%PROJECT_ROOT%\python-3.12.9-embed;%PATH%"

echo Embedded Python: python-3.12.9-embed\
echo Env tools: env\ (sox, wget, etc.)
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
