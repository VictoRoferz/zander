@echo off
setlocal
REM Update the Zander station to the latest code. Run it after new code was
REM pushed (needs internet):
REM   1. git pull --ff-only        stops on local edits or a diverged branch
REM   2. docker compose build      new receiver/dashboard/inspector images
REM   3. stop-zander.bat           the running camera stack holds its packages'
REM                                DLLs open, so pip could not replace them
REM   4. camera stack packages     pip install -r camerapi\requirements.txt
REM   5. start-zander.bat          docker compose up -d with the new images,
REM                                camera stack with the new code
REM If step 1 or 2 fails, nothing after it runs and the station keeps running
REM the version it had. Once the station is stopped (step 3) it is always
REM started again, even if step 4 fails. Every step is appended to
REM deploy\update-zander.log.
REM
REM git pull may replace this very file, and cmd reads a running batch file
REM from disk line by line: the work runs from a copy in %TEMP%, so the pull
REM cannot rewrite the script under its own feet.
if /i "%~1"=="--from-copy" goto :main
copy /y "%~f0" "%TEMP%\zander-update-running.bat" >nul
if errorlevel 1 (
  echo *** Could not copy the update script to the TEMP folder - update not started ***
  pause
  exit /b 1
)
REM Without "call": control moves to the copy and never comes back here
REM (the exit below only runs if the copy could not be started at all).
"%TEMP%\zander-update-running.bat" --from-copy "%~dp0"
exit /b 1

:main
set "DEPLOY=%~2"
cd /d "%DEPLOY%"
set "UZLOG=%DEPLOY%update-zander.log"
set "REPO=%DEPLOY%.."
call :log "=== update-zander invoked ==="

REM --- 1. Code ---
where git >nul 2>&1
if errorlevel 1 goto :no_git
set "OLDREV=unknown"
for /f %%r in ('git -C "%REPO%" rev-parse --short HEAD 2^>nul') do set "OLDREV=%%r"
call :log "Pulling the latest code (now at %OLDREV%)..."
git -C "%REPO%" pull --ff-only
if errorlevel 1 goto :pull_failed
set "NEWREV=unknown"
for /f %%r in ('git -C "%REPO%" rev-parse --short HEAD 2^>nul') do set "NEWREV=%%r"
call :log "Code is at %NEWREV% (was %OLDREV%)."

REM --- 2. Docker images ---
docker info >nul 2>&1
if errorlevel 1 goto :no_docker
call :log "Building the Docker images (docker compose build)..."
docker compose build
if errorlevel 1 goto :build_failed

REM --- 3. Stop: the camera stack only picks up new code when restarted, and
REM its running Python holds the packages' DLLs open (pip cannot replace them) ---
call :log "Stopping Zander for the update..."
REM stop-zander.bat ends with "pause": an empty stdin lets it return at once.
call "%DEPLOY%stop-zander.bat" <nul
cd /d "%DEPLOY%"
REM taskkill returns before Windows has released the killed processes' files.
timeout /t 3 /nobreak >nul

REM --- 4. Camera stack packages (only if the camera stack was set up) ---
set "PIPOK=yes"
set "CAMPY=%DEPLOY%..\.venv\Scripts\python.exe"
if not exist "%CAMPY%" goto :restart
call :log "Updating the camera stack's Python packages..."
"%CAMPY%" -m pip install -r "%REPO%\camerapi\requirements.txt"
if errorlevel 1 set "PIPOK=no"
if "%PIPOK%"=="no" call :log "*** Updating the camera stack's packages FAILED - see above; usually no internet. Starting anyway with the packages it had ***"

:restart
REM --- 5. Start the new version (always, once the station was stopped) ---
call :log "Starting Zander with the new version..."
call "%DEPLOY%start-zander.bat"
if errorlevel 1 goto :start_problem
if "%PIPOK%"=="no" goto :pip_problem
call :log "Update to %NEWREV% complete."
exit /b 0

:start_problem
REM start-zander.bat has already shown its problem and waited for a key.
call :log "*** Updated to %NEWREV%, but start-zander.bat reported a problem - see its messages ***"
exit /b 1

:pip_problem
call :log "*** Updated to %NEWREV%, but the camera stack's packages are not updated. If the Zander Camera window shows an import error, run update-zander.bat again once online ***"
goto :failed

:no_git
call :log "*** git is not installed or not on PATH - install Git for Windows, then run update-zander.bat again ***"
goto :failed

:pull_failed
call :log "*** git pull FAILED - nothing was changed. No internet, or local edits in the repo: run git status in the repo folder ***"
goto :failed

:no_docker
call :log "*** Docker is not running - open Docker Desktop, wait until it says running, then run update-zander.bat again. The code is already pulled; nothing was restarted ***"
goto :failed

:build_failed
call :log "*** docker compose build FAILED - see above; usually no internet. The station keeps running the version it had; run update-zander.bat again once online ***"
goto :failed

:failed
echo.
echo Update not finished - see the message above (log: deploy\update-zander.log).
pause
exit /b 1

:log
echo %~1
>> "%UZLOG%" echo [%date% %time%] %~1
goto :eof
