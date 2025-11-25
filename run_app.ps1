# Set required secrets – replace the placeholder values with your real credentials
$env:VOCHI_CLIENT_ID = 'elq3tE3JhZWuKi8HsGI6PkipkYiUbjJN'
$env:GOOGLE_API_KEY   = 'AIzaSyDpRoX-KRPUGWp-yCSMmUO1heZxJJna76g'
$env:VOCHI_UI_PASSWORD = '123'

# Optional: activate a virtual environment if you have one
# .\venv\Scripts\Activate.ps1   # uncomment if a venv exists

# Run the application
python app.py
