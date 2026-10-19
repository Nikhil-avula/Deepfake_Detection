@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the project virtual environment using INSTALLATION.md first.
  echo Required: py -3.12 -m venv .venv
  echo Then install the documented dependencies in that environment.
  exit /b 1
)
set "PYTHONPATH=%CD%\src"
echo Checking this project and its trained models...
".venv\Scripts\python.exe" -m deepguard.cli doctor --encoder
if errorlevel 1 (
  echo Startup checks failed. Fix the named failed check using INSTALLATION.md.
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app/main.py --server.fileWatcherType none --server.maxUploadSize 100
