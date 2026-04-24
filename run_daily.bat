@echo off
cd /d "d:\googlePRJ\Amedis"
call venv\Scripts\activate.bat
python -m calls_analyser.runner >> daily_log.txt 2>&1
