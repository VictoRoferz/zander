@echo off
REM One-time setup for the native camera stack (camerapi + USB button listener).
REM Creates the repo venv (shared with the windows-native fallback path) and
REM installs camerapi's dependencies (incl. pypylon + pynput).
REM Usage: setup-camera.bat [auto]
REM   "auto" = non-interactive (no pauses); start-zander.bat uses this to
REM   self-heal a missing or half-finished setup on icon start.
set "AUTO="
if /i "%~1"=="auto" set "AUTO=1"
cd /d "%~dp0\.."

if not exist .venv (
  echo Creating Python venv at %CD%\.venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo *** Could not create the venv. Is Python 3.10+ installed and on PATH? ***
    if not defined AUTO pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r camerapi\requirements.txt
if errorlevel 1 (
  echo *** pip install failed - see output above. ***
  if not defined AUTO pause
  exit /b 1
)

if not exist camerapi\.env (
  copy camerapi\.env.example camerapi\.env >nul
  echo Created camerapi\.env from the example ^(defaults suit this single-PC setup^).
)

REM Completion marker checked by start-zander.bat — written only when every
REM step above succeeded, so a half-finished setup gets re-run, not trusted.
echo ok > .venv\camera-stack-ready

echo.
echo Camera stack is set up. Next steps:
echo   1. Program the USB keypad ^(or keep its copy button; see camerapi\.env^).
echo   2. Make sure the camera is reachable: ping its IP.
echo   3. Start everything with the Zander desktop icon ^(or start-zander.bat^).
if not defined AUTO pause
exit /b 0
