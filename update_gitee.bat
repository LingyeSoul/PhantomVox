@echo off
echo ========================================
echo  PhantomVox Updater (Gitee) - Git
echo ========================================
echo.

REM Set Python virtual environment path
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

REM Check if git exists
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found!
    echo Please install Git first.
    pause
    exit /b 1
)

REM Check if Python virtual environment exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python virtual environment not found at %PYTHON_EXE%
    echo Please run setup script first.
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

echo [INFO] Fetching updates from Gitee...
echo.

REM Add Gitee remote if not exists
git remote get-url gitee >nul 2>&1
if errorlevel 1 (
    echo [INFO] Adding Gitee remote...
    git remote add gitee https://gitee.com/lingyesoul/PhantomVox.git
)

REM Fetch updates from Gitee
git fetch gitee

REM Check if there are updates
for /f %%i in ('git rev-parse HEAD') do set LOCAL=%%i
for /f %%i in ('git rev-parse gitee/main') do set REMOTE=%%i

if "%LOCAL%"=="%REMOTE%" (
    echo [INFO] Already up to date!
    echo.
    goto :end
)

echo [INFO] New updates found!

:pull

echo [INFO] Pulling changes...
echo.

REM Pull changes from Gitee (with autostash and rebase)
git pull --rebase --autostash gitee main

echo.
echo ========================================
echo  Code update completed!
echo ========================================
echo.

REM Update dependencies
echo [1/3] Updating dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [2/3] Uninstalling old qwen-tts...
"%PYTHON_EXE%" -m pip uninstall -y qwen-tts

echo [3/3] Installing qwen-tts from Gitee...
"%PYTHON_EXE%" -m pip install git+https://gitee.com/lingyesoul/Qwen3-TTS-Streaming

echo.
echo ========================================
echo  Update completed successfully!
echo ========================================
echo.

:end
pause
