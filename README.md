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

For a tenant on **VoChi API v1**, set:

- `<TENANT>_TELEPHONY_PROVIDER=vochi`
- `<TENANT>_VOCHI_API_KEY=...`
- `<TENANT>_VOCHI_BASE_URL=https://bot.vochi.by/api/v1` (optional)

The main VoChi list is loaded from `/calls` with an explicitly empty
`phone=` query parameter. This returns calls across all phone numbers; the
parameter must be present because omitting it produces HTTP 422.

The application keeps answered calls only:

- `call_status=2` — included;
- `call_status=0` or `call_status=1` — excluded as unsuccessful.

The UI is unchanged. Date, time, and call-type filters continue to apply, and
no phone-number or call-status field is added. The main integration does not
use `/unsuccessful-calls`. Recordings are resolved through `/recording` and
downloaded from the temporary S3 URL returned by the API.

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

## Scheduled batches

`SCHEDULER_TIMEZONE` selects the one IANA timezone used by every tenant
schedule (for example, `Europe/Nicosia`). It defaults to `UTC`. Enabled
tenants and their cron or interval settings are resolved once when the
application starts, so schedule changes require an application restart.

Scheduled execution requires complete `SUPABASE_URL` and `SUPABASE_KEY`
credentials and the `scheduler_runs` schema from
`docs/supabase/multi_tenant_schema.sql` applied to that Supabase project. The
scheduler fails closed when it cannot create or use this run guard; it never
falls back to an unguarded batch.

Each planned tenant slot is claimed once before any calls are listed or
downloaded. An interrupted slot left in `running` state requires manual
intervention. This is intentional: there is no heartbeat, stale-run takeover,
automatic backfill, or dynamic rescheduling.

## Email reports

Batch results can be sent through Brevo HTTPS API, with Gmail SMTP as a fallback, as:

- a filtered HTML table in the email body;
- a UTF-8 CSV attachment containing all batch results.

Configure Brevo as the primary provider:

- `BREVO_API_KEY` — Brevo transactional email API key;
- `EMAIL_FROM` — optional verified Brevo sender address. Defaults to `tuttstt@gmail.com`;
- `EMAIL_FROM_NAME` — optional sender display name. Defaults to no display name;
- `EMAIL_TO` — optional recipient address. Defaults to `tuttstt@gmail.com`.

Configure Gmail fallback:

- `GOOGLE_app` — the Gmail app password for `tuttstt@gmail.com`;
- `EMAIL_TO` — optional recipient address. Defaults to `tuttstt@gmail.com`.

When `BREVO_API_KEY` is set, Brevo is used. If it is absent and `GOOGLE_app`
is set, Gmail is used instead.

The Gradio UI provides a **Send by email** button. The selected batch filter
controls the HTML table, while the CSV remains unfiltered. Scheduled and CLI
daily batches send `Needs follow-up` rows in the HTML table after processing
and attach the complete CSV.
