@echo off
:: 设置项目根目录为当前目录
set PROJECT_ROOT=%cd%
:: 将 src 目录加入 Python 搜索路径
set PYTHONPATH=%PROJECT_ROOT%\src
:: 将 env 文件夹加入 PATH（sox, wget 等工具）
set PATH=%PROJECT_ROOT%\env;%PROJECT_ROOT%\python-3.12.9-embed;%PATH%
:: 运行 main.py
.\python-3.12.9-embed\python.exe -s %PROJECT_ROOT%\src\main.py
:: 运行完暂停，查看输出
pause