@echo off
setlocal
REM Start the Zander station: the Docker hub (Label Studio + receiver +
REM dashboard) AND the native camera stack (camerapi + USB button listener).
REM Double-click this file (or the desktop "Zander" shortcut) to run.
REM Every decision is appended to deploy\start-zander.log for diagnosis.
cd /d "%~dp0"
set "SZLOG=%~dp0start-zander.log"
call :log "=== start-zander invoked ==="

call :log "Starting Zander hub (Docker)..."
docker compose up -d
if errorlevel 1 (
  call :log "*** docker compose up FAILED - is Docker Desktop running? ***"
  echo Open Docker Desktop, wait until it says "running", then try again.
  pause
  exit /b 1
)
call :log "Docker hub is up."

REM --- Native camera stack (self-healing) ---
set "CAMPY=%~dp0..\.venv\Scripts\python.exe"
set "CAMREADY=%~dp0..\.venv\camera-stack-ready"
set "CAMPIDFILE=%~dp0camera-stack.pid"

if exist "%CAMREADY%" goto :camera_check
call :log "Camera stack not set up yet - running the one-time setup (needs internet)..."
call "%~dp0setup-camera.bat" auto
cd /d "%~dp0"
if exist "%CAMREADY%" goto :camera_check
call :log "*** Camera setup FAILED - starting WITHOUT the camera + USB button ***"
echo *** Run deploy\setup-camera.bat by hand to see the error. ***
pause
goto :camera_done

:camera_check
REM The truth about "already running" is the port, not a PID file.
curl -s -f -m 2 http://localhost:8001/api/v1/health >nul 2>&1
if not errorlevel 1 (
  call :log "Camera stack already running (:8001 health check OK)."
  goto :camera_done
)
if not exist "%CAMPIDFILE%" goto :camera_start
set /p CAMPID=<"%CAMPIDFILE%"
call :log "Port 8001 dead but PID file exists - cleaning up stale camera stack (PID %CAMPID%)..."
taskkill /PID %CAMPID% /T /F >nul 2>&1
del "%CAMPIDFILE%" >nul 2>&1

:camera_start
call :log "Starting camera stack (camerapi + USB button)..."
start "Zander Camera" /min "%CAMPY%" "%~dp0..\scripts\launch.py" --camera-only --ignore-ctrl-c --pid-file "%CAMPIDFILE%" --log-file "%~dp0camera-stack.log"

:camera_done
echo Waiting a few seconds for services to come up...
timeout /t 6 /nobreak >nul

call :log "Opening the dashboard in the browser..."
REM Edge is preinstalled on every Windows 10/11; fall back to the default
REM browser association if Edge is unavailable.
start "" msedge "http://localhost:8003" 2>nul || start "" "http://localhost:8003"

echo.
echo Zander is running:
echo   Dashboard        http://localhost:8003
echo   Label Studio     http://localhost:8081
echo   Camera + button  minimized "Zander Camera" window
echo                    (camera log: deploy\camera-stack.log; this script's
echo                     decisions: deploy\start-zander.log)
echo.
echo (You can close this window; the services keep running.)
timeout /t 4 /nobreak >nul
exit /b 0

:log
echo %~1
>> "%SZLOG%" echo [%date% %time%] %~1
goto :eof
