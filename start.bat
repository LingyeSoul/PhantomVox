@echo off
echo ========================================
echo Starting PhantomVox
echo ========================================
echo.

set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0env;%PATH%"

%~dp0python\python.exe src\main.py

echo.
echo ========================================
echo Program exited
echo ========================================
pause
