@echo off
REM 地球生成器 web 服务一键启动
REM 必须使用包含 torch / numba 的 venv 解释器，不要改用裸 python
setlocal
set "PORT=8899"
cd /d "%~dp0"

REM 优先使用 WorkBuddy 托管环境；否则回退到项目本地 .venv-torch（含 torch+cu128，可用 CUDA）
set "PY=C:/Users/Qq203/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
if not exist "%PY%" set "PY=%~dp0.venv-torch\Scripts\python.exe"

if not exist "%PY%" (
  echo [错误] 找不到可用的 venv 解释器：
  echo   1. WorkBuddy: C:/Users/Qq203/.workbuddy/binaries/python/envs/default/Scripts/python.exe
  echo   2. 本地:      %~dp0.venv-torch\Scripts\python.exe
  echo 请创建其一（.venv-torch 需安装 torch/flask/numba/h5py/scipy/matplotlib/numpy）。
  pause
  exit /b 1
)

echo 使用解释器: %PY%
echo 正在启动地球生成器 (http://127.0.0.1:%PORT%/) ...
"%PY%" tools/web_server.py --port %PORT%
echo.
echo [服务已停止] 按任意键关闭窗口。
pause
