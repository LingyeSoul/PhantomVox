@echo off
REM 设置 UTF-8 编码以正确显示中文
chcp 65001 >nul 2>&1

REM PhantomVox 更新器 - Gitee 版本
REM 使用 Gitee 仓库进行更新（国内用户推荐）

echo ========================================
echo  PhantomVox Updater (Gitee)
echo ========================================
echo.

REM 设置 Python 环境
set "PYTHON_EXE=%~dp0python\python.exe"
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0env;%PATH%"

REM 检查 Python 是否存在
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到 Python 环境
    echo 路径: %PYTHON_EXE%
    echo.
    echo 请先运行 setup_env.bat 安装依赖
    echo.
    pause
    exit /b 1
)

REM 定义仓库 URL（Gitee）
set "REPO_URL=https://gitee.com/lingyesoul/PhantomVox"

REM 复制更新脚本到根目录
echo [信息] 正在准备更新脚本...
copy "%~dp0src\update.py" "%~dp0update.py" /Y >nul 2>&1

if not exist "%~dp0update.py" (
    echo [错误] 复制更新脚本失败
    echo 请确保 src\update.py 文件存在
    echo.
    pause
    exit /b 1
)

REM 执行更新
echo [信息] 开始检查更新...
echo.
echo 仓库地址: %REPO_URL%
echo.
"%PYTHON_EXE%" "%~dp0update.py" "%REPO_URL%"

REM 清理临时文件
if exist "%~dp0update.py" del "%~dp0update.py" >nul 2>&1

echo.
echo ========================================
echo  更新程序已退出
echo ========================================
echo.
pause
