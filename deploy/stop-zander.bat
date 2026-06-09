@echo off
REM Stop the Zander laptop hub. Data is preserved (volumes + the data folder).
cd /d "%~dp0"
echo Stopping Zander hub...
docker compose down
echo Zander stopped. (Images, the data folder, and Label Studio's DB are kept.)
pause
