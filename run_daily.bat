@echo off
cd /d "d:\googlePRJ\Amedis"
:: Adjust the path to python if it's not in your global PATH, e.g., C:\Python39\python.exe
python run_daily_batch.py >> daily_log.txt 2>&1
