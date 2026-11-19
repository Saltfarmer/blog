@echo off
rem Double-click to run this site locally and open it in the browser.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_serve.ps1" %*
pause
