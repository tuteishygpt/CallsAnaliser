
import os
from dotenv import find_dotenv, load_dotenv

print(f"CWD: {os.getcwd()}")
env_path = find_dotenv()
print(f"Found .env at: {env_path}")

if env_path:
    load_dotenv(env_path, override=True)
    url = os.environ.get("SUPABASE_URL")
    print(f"SUPABASE_URL present: {bool(url)}")
    if url:
        print(f"SUPABASE_URL length: {len(url)}")
        print(f"SUPABASE_URL starts with 'https://': {url.startswith('https://')}")
else:
    print(".env file not found!")
