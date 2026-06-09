@echo off
REM Start the Zander laptop hub (Label Studio + receiver + dashboard) and open the dashboard.
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

echo Waiting a few seconds for services to come up...
timeout /t 6 /nobreak >nul

echo Opening the dashboard...
start "" http://localhost:8003

echo.
echo Zander is running:
echo   Dashboard      http://localhost:8003
echo   Label Studio   http://localhost:8081
echo.
echo (You can close this window; the services keep running in Docker.)
timeout /t 4 /nobreak >nul
