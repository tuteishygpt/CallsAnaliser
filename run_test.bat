@echo off
cd /d "d:\googlePRJ\Amedis"

:: Прыклад тэставага запуску (па змаўчанню бярэцца ўчорашні дзень)
:: Фільтр па часе з 12:00 да 14:00
python run_daily_batch.py --time-from 12:00 --time-to 14:00

:: Каб запусціць для канкрэтнай даты, раскаментуйце і змяніце дату:
:: python run_daily_batch.py --date 2025-12-10 --time-from 09:00 --time-to 10:00

pause
