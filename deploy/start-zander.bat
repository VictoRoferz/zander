@echo off
setlocal
REM Start the Zander station: the Docker hub (Label Studio + receiver +
REM dashboard + inspector) AND the native camera stack (camerapi + USB button
REM listener). Double-click this file (or the desktop "Zander" shortcut) to run.
REM Every decision is appended to deploy\start-zander.log for diagnosis.
REM
REM The camera stack ALWAYS starts, whatever happens to the hub: captures are
REM spooled locally and uploaded once the hub is up, so a Docker or image-build
REM problem never stops the station from capturing.
cd /d "%~dp0"
set "SZLOG=%~dp0start-zander.log"
set "HUB=up"
set "CAM=down"
call :log "=== start-zander invoked ==="

REM --- Docker hub ---
REM Docker itself first, so a build error is never reported as "Docker is not running".
docker info >nul 2>&1
if errorlevel 1 goto :docker_down

REM --build rebuilds the receiver/dashboard/inspector images after a git pull.
REM Unchanged images come from the layer cache, so this also works offline.
call :log "Starting Zander hub (Docker; rebuilding changed images)..."
docker compose up -d --build
if not errorlevel 1 goto :hub_done

REM A failed build makes docker compose start NO container at all.
set "HUB=stale"
call :log "*** docker compose up --build FAILED (usually an image build that needs internet, see above) - starting the existing images instead ***"
docker compose up -d
if not errorlevel 1 goto :hub_done

REM Typically the inspector image was never built: its first build needs internet.
set "HUB=degraded"
call :log "*** Starting the existing images FAILED too - starting the core services without the inspector ***"
docker compose up -d labelstudio receiver dashboard
if not errorlevel 1 goto :hub_done

set "HUB=failed"
call :log "*** The core services failed to start as well - see the messages above ***"
goto :hub_done

:docker_down
set "HUB=nodocker"
call :log "*** Docker is not running (or not installed) - starting only the camera stack ***"

:hub_done
if "%HUB%"=="up" call :log "Docker hub is up."
if "%HUB%"=="stale" call :log "Docker hub is up with the images built before - after a code update that is the OLD version (build failed - needs internet?)"
if "%HUB%"=="degraded" call :log "DEGRADED: inspector not running (build failed - needs internet once)"

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
  set "CAM=running"
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
set "CAM=running"

:camera_done
echo Waiting a few seconds for services to come up...
timeout /t 6 /nobreak >nul

REM No dashboard without the hub: skip the browser then.
if "%HUB%"=="nodocker" goto :summary
if "%HUB%"=="failed" goto :summary
call :log "Opening the dashboard in the browser..."
REM Edge is preinstalled on every Windows 10/11; fall back to the default
REM browser association if Edge is unavailable.
start "" msedge "http://localhost:8003" 2>nul || start "" "http://localhost:8003"

:summary
if not "%HUB%"=="up" goto :summary_problem
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

:summary_problem
echo.
echo ******************************************************************
if "%HUB%"=="stale" echo   The images could not be rebuilt - no internet? The containers run the
if "%HUB%"=="stale" echo   images built before: after a code update that is the OLD version.
if "%HUB%"=="stale" echo   Connect to the internet and run update-zander.bat.
if "%HUB%"=="degraded" echo   DEGRADED: the inspector - the defect detection - is not running.
if "%HUB%"=="degraded" echo   Its image was never built: that needs internet once. Connect and
if "%HUB%"=="degraded" echo   run start-zander.bat again. Capture and labeling work meanwhile.
if "%HUB%"=="failed" echo   The Docker containers did not start - see the messages above.
if "%HUB%"=="nodocker" echo   Docker is not running: open Docker Desktop, wait until it says
if "%HUB%"=="nodocker" echo   "running", then start Zander again.
if "%CAM%"=="running" if "%HUB%"=="nodocker" echo   The camera stack runs: captures are kept until the hub is up.
if "%CAM%"=="running" if "%HUB%"=="failed" echo   The camera stack runs: captures are kept until the hub is up.
echo   Details: deploy\start-zander.log
echo ******************************************************************
pause
exit /b 1

:log
echo %~1
>> "%SZLOG%" echo [%date% %time%] %~1
goto :eof
