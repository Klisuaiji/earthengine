@echo off
REM 地球生成器 web 服务一键启动
REM 必须使用包含 torch / numba 的 venv 解释器，不要改用裸 python
set "PY=C:/Users/Qq203/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
set "PORT=8899"
cd /d "%~dp0"
if not exist "%PY%" (
  echo [错误] 找不到 venv 解释器：%PY%
  echo 请确认 WorkBuddy 的 envs/default 环境已创建。
  pause
  exit /b 1
)
echo 正在启动地球生成器 (http://127.0.0.1:%PORT%/) ...
"%PY%" tools/web_server.py --port %PORT%
echo.
echo [服务已停止] 按任意键关闭窗口。
pause
