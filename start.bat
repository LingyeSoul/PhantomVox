@echo off
echo ========================================
echo Starting PhantomVox
echo ========================================
echo.

REM Set environment
set "PATH=%~dp0env;%PATH%"

python\python.exe src\main.py

echo.
echo ========================================
echo Program exited
echo ========================================
pause
