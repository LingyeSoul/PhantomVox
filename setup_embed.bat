@echo off
echo ========================================
echo PhantomVox Environment Setup (Embed Python)
echo ========================================
echo.

set PYTHON_EXE=%~dp0python-3.12.9-embed\python.exe

REM Check if embed Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python embed not found at %PYTHON_EXE%
    echo Please ensure python-3.12.9-embed directory exists.
    pause
    exit /b 1
)

echo [1/4] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo [2/4] Installing PyTorch 2.9
"%PYTHON_EXE%" -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

echo [3/4] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"

echo [3.5/4] Installing qwen-tts from GitHub...
"%PYTHON_EXE%" -m pip install git+https://github.com/LingyeSoul/Qwen3-TTS-Streaming

echo [4/4] Installing flash-attention...
"%PYTHON_EXE%" -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.6/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-win_amd64.whl

echo ========================================
echo Setup completed!
echo ========================================
echo.
echo Installed versions:
"%PYTHON_EXE%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo.
pause
