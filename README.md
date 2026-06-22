---
title: lix
emoji: 📞
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
---

# Calls Analyser

A hexagonal architecture Python project for analysing and summarising call transcripts.

## Gemini BATCH mode

To enable Gemini BATCH processing for mass analysis, adjust `batch_params.json` in the project root:

```json
{
  "enable_gemini_batch": true,
  "batch_size": 25
}
```

- `enable_gemini_batch`: set to `true` to send batch jobs through the Gemini BATCH API instead of per-call requests.
- `batch_size`: how many recordings are packed into a single Gemini BATCH job (minimum 1).

## Tenant provider configuration

By default the app uses **Vochi** provider (`TELEPHONY_PROVIDER=vochi`).

For a new tenant on **MTS VATS**, set tenant-scoped environment variables:

- `<TENANT>_TELEPHONY_PROVIDER=mts_vats`
- `<TENANT>_MTS_DOMAIN=193130978.vats.mts.by`
- `<TENANT>_MTS_API_KEY=...`

Field mapping used by MTS VATS integration:

- `uid` → `CallLogEntry.unique_id`
- `start` → `CallLogEntry.started_at`
- `client` → `CallLogEntry.caller_id`
- `destination` → `CallLogEntry.destination`
- `duration` → `CallLogEntry.duration_seconds`
- `record`/`history/record/{uid}` → recording download URL

## Hugging Face Spaces deployment

This Space requires the following Secrets (Settings → Secrets and variables):

- `GOOGLE_SERVICE_ACCOUNT_JSON_B64` — base64-encoded GCP service account JSON
- `GOOGLE_CLOUD_PROJECT` — your GCP project id
- `GOOGLE_CLOUD_LOCATION` — Vertex location (`global` works)
- `DEFAULT_TENANT_ID` — e.g. `lix`
- `LIX_TELEPHONY_PROVIDER` — `mts_vats`
- `LIX_MTS_DOMAIN` — e.g. `193991078.vats.mts.by`
- `LIX_MTS_API_KEY` — MTS VATS API key
- `VOCHI_UI_PASSWORD` — UI password

Optional: `SUPABASE_URL` + `SUPABASE_KEY` to use Supabase as cache.
