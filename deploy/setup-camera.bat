@echo off
REM One-time setup for the native camera stack (camerapi + USB button listener).
REM Creates the repo venv (shared with the windows-native fallback path) and
REM installs camerapi's dependencies (incl. pypylon + pynput).
cd /d "%~dp0\.."

if not exist .venv (
  echo Creating Python venv at %CD%\.venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo *** Could not create the venv. Is Python 3.10+ installed and on PATH? ***
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r camerapi\requirements.txt
if errorlevel 1 (
  echo *** pip install failed - see output above. ***
  pause
  exit /b 1
)

if not exist camerapi\.env (
  copy camerapi\.env.example camerapi\.env >nul
  echo Created camerapi\.env from the example ^(defaults suit this single-PC setup^).
)

echo.
echo Camera stack is set up. Next steps:
echo   1. Program the USB keypad: key 1 = F13 ^(capture^), key 2 = F14 ^(test shot^).
echo   2. Give the camera NIC a static IP on the camera subnet ^(192.168.177.x^).
echo   3. Start everything with the Zander desktop icon ^(or start-zander.bat^).
pause
