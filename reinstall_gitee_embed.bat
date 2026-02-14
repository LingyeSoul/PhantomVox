@echo off
echo ========================================
echo  PhantomVox Reinstaller (Gitee) - Embed
echo ========================================
echo.
echo This script will reinstall all dependencies.
echo Use this when encountering dependency issues.
echo.

REM Set Python path
set "PYTHON_EXE=%~dp0python-3.12.9-embed\python.exe"

REM Check if embedded Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python embed not found at %PYTHON_EXE%
    echo Please ensure python-3.12.9-embed directory exists.
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
