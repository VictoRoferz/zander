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

REM --- Native camera stack (self-healing: runs the setup itself if needed) ---
set "CAMPY=%~dp0..\.venv\Scripts\python.exe"
set "CAMREADY=%~dp0..\.venv\camera-stack-ready"
set "CAMPIDFILE=%~dp0camera-stack.pid"

if exist "%CAMREADY%" goto :camera_check
echo Camera stack not set up yet - running the one-time setup now (needs internet)...
call "%~dp0setup-camera.bat" auto
cd /d "%~dp0"
if exist "%CAMREADY%" goto :camera_check
echo.
echo *** Camera setup FAILED - starting WITHOUT the camera + USB button.  ***
echo *** Run deploy\setup-camera.bat by hand to see the error.            ***
pause
goto :camera_done

:camera_check
if not exist "%CAMPIDFILE%" goto :camera_start
set /p CAMPID=<"%CAMPIDFILE%"
REM Only trust the PID if it is actually a python process — after a reboot
REM Windows may hand a stale PID to an unrelated program.
tasklist /FI "PID eq %CAMPID%" /FI "IMAGENAME eq python.exe" 2>nul | find "%CAMPID%" >nul
if not errorlevel 1 (
  echo Camera stack already running (PID %CAMPID%).
  goto :camera_done
)

:camera_start
echo Starting camera stack (camerapi + USB button)...
start "Zander Camera" /min "%CAMPY%" "%~dp0..\scripts\launch.py" --camera-only --ignore-ctrl-c --pid-file "%CAMPIDFILE%" --log-file "%~dp0camera-stack.log"

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
