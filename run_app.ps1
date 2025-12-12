# Set required secrets – replace the placeholder values with your real credentials
$env:VOCHI_CLIENT_ID = 'elq3tE3JhZWuKi8HsGI6PkipkYiUbjJN'
$env:GOOGLE_API_KEY = 'AIzaSyDCHLf8iRsi8IeIPL7SsXYSVP77m3IYL_M'
$env:VOCHI_UI_PASSWORD = '123'
$env:DATABASE_URL = 'https://funxpzdaazscyyjwybch.supabase.co'
$env:SUPABASE_KEY = 'sb_secret_OCgSV2V9xYiZWVry5pwB5w_ujAcEIZQ'

# Optional: activate a virtual environment if you have one
# .\venv\Scripts\Activate.ps1   # uncomment if a venv exists

# Run the application
python app.py
