@echo off
REM Start all three services without Docker, via scripts\launch.py.
REM Reads the Label Studio token from receiver\.env and dashboard\.env.
setlocal
cd /d "%~dp0\..\.."
REM now in the zander\ repo root

if not exist ".venv\Scripts\activate.bat" (
  echo .venv not found - run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

if "%DATA_ROOT%"=="" set DATA_ROOT=%USERPROFILE%\zander-data
echo DATA_ROOT=%DATA_ROOT%
echo Starting Label Studio (8081) + receiver (8002) + dashboard (8003)...
echo Press Ctrl-C in this window to stop everything.

python scripts\launch.py
pause
