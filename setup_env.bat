@echo off
echo ========================================
echo PhantomVox Environment Setup (System VEnv)
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please ensure Python is installed and in PATH.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call "%~dp0.venv\Scripts\activate.bat"

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo [2/4] Installing PyTorch 2.9
python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

echo [3/4] Installing dependencies...
python -m pip install -r requirements.txt

echo [3.5/4] Installing local qwen-tts wheel...
python -m pip install "%~dp0qwen_tts-0.0.6-py3-none-any.whl"

echo [4/4] Installing flash-attention...
python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.6/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-win_amd64.whl

echo ========================================
echo Setup completed!
echo ========================================
echo.
echo Installed versions:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo.
echo ========================================
echo Please download and install SoX from https://sourceforge.net/projects/sox/
echo qwen-tts require sox
echo ========================================
echo Usage:
echo   Run start_venv.bat to launch the program
echo   Run cmd_venv.bat for command line environment
echo.
pause
