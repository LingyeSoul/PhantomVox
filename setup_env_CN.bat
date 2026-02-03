@echo off
echo ========================================
echo PhantomVox Environment Setup
echo ========================================

set PYTHON_EXE=python\python.exe
set "PATH=%~dp0env;%PATH%"

echo [1/4] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo [2/4] Installing PyTorch 2.9
"%PYTHON_EXE%" -m pip install torch==2.9.1 torchaudio==2.9.1  --index-url https://download.pytorch.org/whl/cu128

echo [3/4] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo [4/4] Installing flash-attention...
"%PYTHON_EXE%" -m pip install https://v6.gh-proxy.org/https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.6/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-win_amd64.whl

echo ========================================
echo Setup completed!
echo ========================================
echo.
echo Installed versions:
"%PYTHON_EXE%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo.
echo Usage:
echo   Run start.bat to launch the program
echo   Run cmd.bat for command line environment
echo.
pause
