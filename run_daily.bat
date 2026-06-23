@echo off
pushd "%~dp0"
"venv\Scripts\python.exe" -m calls_analyser.runner >> "daily_log.txt" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
