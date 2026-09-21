@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-client.ps1" -Conda %*
set "ROWAN_EXIT=%ERRORLEVEL%"
if not "%ROWAN_EXIT%"=="0" pause
exit /b %ROWAN_EXIT%
