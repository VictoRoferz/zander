@echo off
REM One-time setup for the no-Docker path: create a venv and install everything.
REM Requires Python 3.11 on PATH.
setlocal
cd /d "%~dp0\..\.."
REM now in the zander\ repo root

echo Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 ( echo Could not create venv - is Python 3.11 installed and on PATH? & pause & exit /b 1 )

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing Label Studio (this can take several minutes)...
pip install label-studio
echo Installing receiver + dashboard dependencies...
pip install -r receiver\requirements.txt
pip install -r dashboard\requirements.txt

echo.
echo Setup complete.
echo Next: put your Label Studio legacy token in receiver\.env and dashboard\.env,
echo then run start.bat in this folder.
pause
