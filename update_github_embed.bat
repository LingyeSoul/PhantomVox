@echo off
echo ========================================
echo  PhantomVox Updater (GitHub) - Embed
echo ========================================
echo.

REM Set git path
set "GIT_EXE=%~dp0env\cmd\git.exe"

REM Check if embedded git exists
if not exist "%GIT_EXE%" (
    echo ERROR: Git not found in env folder!
    echo Path: %GIT_EXE%
    pause
    exit /b 1
)

REM Check if this is a git repository
if not exist "%~dp0.git" (
    echo ERROR: Not a git repository!
    echo Please clone the repository first.
    pause
    exit /b 1
)

echo [INFO] Git: env\cmd\git.exe
echo [INFO] Fetching updates from GitHub...
echo.

REM Fetch updates from remote
"%GIT_EXE%" fetch origin

REM Check if there are updates
for /f %%i in ('"%GIT_EXE%" rev-parse HEAD') do set LOCAL=%%i
for /f %%i in ('"%GIT_EXE%" rev-parse origin/main') do set REMOTE=%%i

if "%LOCAL%"=="%REMOTE%" (
    echo [INFO] Already up to date!
    echo.
    goto :end
)

echo [INFO] New updates found!
echo [INFO] Pulling changes...
echo.

REM Pull changes
"%GIT_EXE%" pull origin main

echo.
echo ========================================
echo  Update completed successfully!
echo ========================================
echo.

:end
pause
