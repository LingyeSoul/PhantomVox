@echo off
echo ========================================
echo PhantomVox Command Environment
echo ========================================
echo.

set "PYTHON_HOME=%~dp0python"
set "PYTHONPATH=%~dp0python\Lib;%~dp0python\Lib\site-packages"
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0env;%PATH%"

echo Python environment activated :
echo   Python:    python\python.exe
echo   Bin tools: env\ (sox.exe, etc)
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
