@echo off
echo ========================================
echo PhantomVox Environment Setup
echo ========================================

set PYTHON_EXE=python\python.exe
set "PATH=%~dp0env;%PATH%"

echo [1/3] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo [2/3] Installing PyTorch 2.9 
"%PYTHON_EXE%" -m pip install torch==2.9.1 torchaudio==2.9.1  --index-url https://download.pytorch.org/whl/cu128

echo [3/3] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt

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
