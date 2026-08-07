@echo off
REM Stop the Zander station: the native camera stack first, then the Docker
REM hub. Data is preserved (volumes + the data folder).
cd /d "%~dp0"

if not exist camera-stack.pid goto :docker
set /p CAMPID=<camera-stack.pid
echo Stopping camera stack (PID %CAMPID%)...
REM /T kills the whole tree (launch.py -> camerapi + button listener). A hard
REM kill is safe: spool writes are atomic and ingest is idempotent.
taskkill /PID %CAMPID% /T /F >nul 2>&1
del camera-stack.pid >nul 2>&1

:docker
echo Stopping Zander hub...
docker compose down
echo Zander stopped. (Images, the data folder, and Label Studio's DB are kept.)
pause
