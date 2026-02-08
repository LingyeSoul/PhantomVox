@echo off
echo ========================================
echo  PhantomVox Updater (GitHub) - Embed
echo ========================================
echo.

REM Set git path
set "GIT_EXE=%~dp0env\cmd\git.exe"

REM Set Python path
set "PYTHON_EXE=%~dp0python-3.12.9-embed\python.exe"

REM Check if embedded git exists
if not exist "%GIT_EXE%" (
    echo ERROR: Git not found in env folder!
    echo Path: %GIT_EXE%
    pause
    exit /b 1
)

REM Check if embedded Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python embed not found at %PYTHON_EXE%
    echo Please ensure python-3.12.9-embed directory exists.
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
echo [INFO] Cleaning untracked files...
echo [WARNING] This will DELETE untracked files!
echo [INFO] Important directories will be preserved (vocal, output, models)
echo.

REM Show what would be deleted (dry-run)
"%GIT_EXE%" clean -fd --dry-run --exclude=vocal --exclude=output --exclude=models
echo.

set /p CONFIRM="Continue with cleanup? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo [INFO] Cleanup cancelled by user
    echo [INFO] Proceeding with pull anyway...
    goto :pull
)

REM Remove untracked files with exclusions
"%GIT_EXE%" clean -fd --exclude=vocal --exclude=output --exclude=models

:pull

echo [INFO] Pulling changes...
echo.

REM Pull changes (with autostash and rebase)
"%GIT_EXE%" pull --rebase --autostash origin main

echo.
echo ========================================
echo  Code update completed!
echo ========================================
echo.

REM Update dependencies
echo [1/2] Updating dependencies...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"

echo [2/2] Updating qwen-tts from GitHub...
"%PYTHON_EXE%" -m pip install git+https://github.com/LingyeSoul/Qwen3-TTS-Streaming

echo.
echo ========================================
echo  Update completed successfully!
echo ========================================
echo.

:end
pause
