@echo off
echo ========================================
echo  PhantomVox Updater (Gitee) - Git
echo ========================================
echo.

REM Check if git exists
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found!
    echo Please install Git first.
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
echo [INFO] Cleaning untracked files...
echo [WARNING] This will DELETE untracked files!
echo [INFO] Important directories will be preserved (vocal, output, models)
echo.

REM Show what would be deleted (dry-run)
git clean -fd --dry-run --exclude=vocal --exclude=output --exclude=models
echo.

set /p CONFIRM="Continue with cleanup? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo [INFO] Cleanup cancelled by user
    echo [INFO] Proceeding with pull anyway...
    goto :pull
)

REM Remove untracked files with exclusions
git clean -fd --exclude=vocal --exclude=output --exclude=models

:pull

echo [INFO] Pulling changes...
echo.

REM Pull changes from Gitee (with autostash and rebase)
git pull --rebase --autostash gitee main

echo.
echo ========================================
echo  Update completed successfully!
echo ========================================
echo.

:end
pause
