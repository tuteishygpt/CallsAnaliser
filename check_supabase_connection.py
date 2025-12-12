import os
import sys
from dotenv import load_dotenv

load_dotenv()
from supabase import create_client, Client

def check_connection():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    print(f"Checking Supabase Connection...")
    print(f"URL: {url}")
    print(f"KEY: {'*' * 4 + key[-4:] if key else 'None'}")

    if not url or "your supabase project" in url.lower():
        print("❌ Error: SUPABASE_URL appears to be missing or is still a placeholder.")
        print("Please edit your configuration to include the actual Supabase URL.")
        return

    if not key or "service_role" in key.lower():
        print("❌ Error: SUPABASE_KEY appears to be missing or is still a placeholder.")
        print("Please edit your configuration to include the actual Supabase Key.")
        return

    try:
        client: Client = create_client(url, key)
        # Try to select from the table, limit 1 just to check permission/existence
        response = client.table("analysis_results").select("*", count="exact").limit(1).execute()
        print("✅ Connection Successful!")
        print(f"Found {response.count} records in 'analysis_results'.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        # Hint about table creation if it's a 404 or similar
        if "relation" in str(e) and "does not exist" in str(e):
             print("\nHint: Did you create the 'analysis_results' table in Supabase?")

if __name__ == "__main__":
    check_connection()
