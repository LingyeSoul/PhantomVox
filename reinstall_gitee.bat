@echo off
echo ========================================
echo  PhantomVox Reinstaller (Gitee) - Git
echo ========================================
echo.
echo This script will reinstall all dependencies.
echo Use this when encountering dependency issues.
echo.

REM Set Python virtual environment path
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

REM Check if Python virtual environment exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python virtual environment not found at %PYTHON_EXE%
    echo Please run setup script first.
    pause
    exit /b 1
)

echo [INFO] Starting reinstallation...
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
echo  Reinstallation completed successfully!
echo ========================================
echo.
pause
