import json
import os
import sys
from supabase import create_client, Client
from pathlib import Path

# Setup Supabase client from env vars (ensure they are set before running)
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("❌ Error: SUPABASE_URL or SUPABASE_KEY environment variables not set.")
    sys.exit(1)

supabase: Client = create_client(url, key)

CACHE_FILE = Path(".cache/analysis_cache.json")

def migrate_cache():
    if not CACHE_FILE.exists():
        print(f"❌ Cache file not found: {CACHE_FILE}")
        return

    print(f"Reading cache from {CACHE_FILE}...")
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return

    print(f"Found {len(data)} items. migrating to Supabase...")
    
    success_count = 0
    error_count = 0
    skipped_count = 0

    for key_json, result in data.items():
        try:
            # Parse the key to extract components
            # Key format: ["tenant", "uid", "prompt_key", "provider", "model", "custom_fragment"]
            try:
                key_parts = json.loads(key_json)
                if not isinstance(key_parts, list) or len(key_parts) != 6:
                    raise ValueError("Invalid key format")
            except Exception:
                # Fallback for old/weird keys? or skip
                print(f"⚠️ Skipping invalid key: {key_json[:50]}...")
                skipped_count += 1
                continue

            tenant_id = key_parts[0]
            call_unique_id = key_parts[1]
            prompt_key = key_parts[2]
            provider_name = key_parts[3]
            model_key = key_parts[4]
            custom_fragment = key_parts[5]

            # Prepare payload
            payload = {
                "tenant_id": tenant_id,
                "call_unique_id": call_unique_id,
                "prompt_key": prompt_key,
                "provider_name": provider_name,
                "model_key": model_key,
                "custom_fragment": custom_fragment,
                "result_text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
            }

            # Upsert into Supabase
            supabase.table("analysis_results").upsert(
                payload, 
                on_conflict="tenant_id, call_unique_id, prompt_key, provider_name, model_key, custom_fragment"
            ).execute()
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"Migrated {success_count} records...", end="\r")

        except Exception as e:
            print(f"\n❌ Failed to insert record for {key_json[:30]}...: {e}")
            error_count += 1

    print(f"\nMigration complete.")
    print(f"✅ Successfully migrated: {success_count}")
    print(f"❌ Failures: {error_count}")
    print(f"⚠️ Skipped (invalid format): {skipped_count}")

if __name__ == "__main__":
    migrate_cache()
