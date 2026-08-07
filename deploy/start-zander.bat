@echo off
setlocal
REM Start the Zander station: the Docker hub (Label Studio + receiver +
REM dashboard) AND the native camera stack (camerapi + USB button listener).
REM Double-click this file (or the desktop "Zander" shortcut) to run.
cd /d "%~dp0"

echo Starting Zander hub (Docker)...
docker compose up -d
if errorlevel 1 (
  echo.
  echo *** Could not start. Is Docker Desktop running? ***
  echo Open Docker Desktop, wait until it says "running", then try again.
  pause
  exit /b 1
)

REM --- Native camera stack (needs the venv from deploy\setup-camera.bat) ---
set "CAMPY=%~dp0..\.venv\Scripts\python.exe"
set "CAMPIDFILE=%~dp0camera-stack.pid"

if not exist "%CAMPY%" (
  echo [warn] Camera venv missing - run deploy\setup-camera.bat once.
  echo [warn] Starting WITHOUT the camera + USB button.
  goto :camera_done
)

if not exist "%CAMPIDFILE%" goto :camera_start
set /p CAMPID=<"%CAMPIDFILE%"
tasklist /FI "PID eq %CAMPID%" 2>nul | find "%CAMPID%" >nul
if not errorlevel 1 (
  echo Camera stack already running (PID %CAMPID%).
  goto :camera_done
)

:camera_start
echo Starting camera stack (camerapi + USB button)...
start "Zander Camera" /min "%CAMPY%" "%~dp0..\scripts\launch.py" --camera-only --pid-file "%CAMPIDFILE%" --log-file "%~dp0camera-stack.log"

:camera_done
echo Waiting a few seconds for services to come up...
timeout /t 6 /nobreak >nul

echo Opening the dashboard...
start "" http://localhost:8003

echo.
echo Zander is running:
echo   Dashboard        http://localhost:8003
echo   Label Studio     http://localhost:8081
echo   Camera + button  minimized "Zander Camera" window
echo                    (logs also in deploy\camera-stack.log)
echo.
echo (You can close this window; the services keep running.)
timeout /t 4 /nobreak >nul
