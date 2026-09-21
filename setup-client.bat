@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-client.ps1"
set "ROWAN_EXIT=%ERRORLEVEL%"
if not "%ROWAN_EXIT%"=="0" pause
exit /b %ROWAN_EXIT%
