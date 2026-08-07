@echo off
REM Stop the Zander station: the native camera stack first, then the Docker
REM hub. Data is preserved (volumes + the data folder).
cd /d "%~dp0"

if not exist camera-stack.pid goto :port_cleanup
set /p CAMPID=<camera-stack.pid
echo Stopping camera stack (PID %CAMPID%)...
REM /T kills the whole tree (launch.py -> camerapi + button listener). A hard
REM kill is safe: spool writes are atomic and ingest is idempotent.
taskkill /PID %CAMPID% /T /F >nul 2>&1
del camera-stack.pid >nul 2>&1

:port_cleanup
REM Belt and braces: also kill anything still holding port 8001 (e.g. a
REM manually started camera stack that has no PID file).
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do taskkill /PID %%p /T /F >nul 2>&1

echo Stopping Zander hub...
docker compose down
echo Zander stopped. (Images, the data folder, and Label Studio's DB are kept.)
pause
