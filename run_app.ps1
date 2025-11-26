# Set required secrets – replace the placeholder values with your real credentials
$env:VOCHI_CLIENT_ID = 'el '
$env:GOOGLE_API_KEY = 'AIzaS '
$env:VOCHI_UI_PASSWORD = '123'

# Optional: activate a virtual environment if you have one
# .\venv\Scripts\Activate.ps1   # uncomment if a venv exists

# Run the application
python app.py
