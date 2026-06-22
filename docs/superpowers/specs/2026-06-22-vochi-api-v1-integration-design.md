# VoChi API v1 Integration Design

## Goal

Replace the legacy `crm.vochi.by/api/calllogs` integration with the new
`bot.vochi.by/api/v1` API. The application must list answered calls across
all phone numbers for a selected day and download available recordings for
playback and AI analysis.

## Scope

The change applies only to the `vochi` telephony provider. The MTS VATS
adapter and its configuration remain unchanged.

The existing `TelephonyPort` interface remains unchanged so the call-log
service, analysis service, CLI, scheduler, and Gradio handlers can continue
to consume `list_calls()` and `get_recording()`.

## Configuration

VoChi tenants use:

- `VOCHI_BASE_URL`, optional, defaulting to `https://bot.vochi.by/api/v1`
- `VOCHI_API_KEY`, required and tenant-scoped when a tenant prefix is used

Legacy `VOCHI_CLIENT_ID` and `VOCHI_BEARER` are no longer required for VoChi.
The API key is sent only as the `key` query parameter. It must not be logged,
embedded in generated links, committed to source control, or included in
error messages.

`TenantConfig` keeps provider-neutral compatibility fields required by
existing consumers, but gains a VoChi API-key field. Its `recording_url()`
method returns the permanent recording endpoint URL without the secret query
parameter for VoChi.

## Listing Calls

`VochiTelephonyAdapter.list_calls()` requests:

`GET {base_url}/calls`

with these query parameters:

- `phone`: an explicitly supplied empty string, which makes the current VoChi
  API return calls across all phone numbers
- `key`: configured API key
- `date_from`: selected day in `YYYY-MM-DD`
- `date_to`: the same selected day
- `limit`: `50`
- `offset`: starts at `0` and increases by the number of returned calls

The `phone` query parameter must be present even though its value is empty.
Omitting it causes the API to return HTTP 422. This all-number behavior is
not documented by VoChi, so adapter tests must explicitly protect the request
format.

The adapter follows pagination until one of these conditions is met:

- the number of collected calls reaches `total`;
- a page contains fewer than `limit` calls;
- a page is empty.

This prevents an infinite loop if the server returns inconsistent pagination
metadata.

After all pages are collected, the adapter excludes unsuccessful calls:

- `call_status=0` -> excluded
- `call_status=1` -> excluded
- `call_status=2` -> included as answered

The UI is not changed and no new phone-number or call-status controls are
added. Existing date, time, and call-type controls continue to work. The
call-type filter is applied client-side to the `call_type` field after the
adapter has fetched the complete result set.

Each API call is mapped to `CallLogEntry`:

- `unique_id` -> `unique_id`
- `start_time` -> `started_at`, parsed as ISO datetime when valid
- `phone_number` -> `caller_id`
- participant extensions, joined with commas -> `destination`
- `duration_seconds` -> `duration_seconds`
- the unchanged API item -> `raw`

Items without a usable `unique_id` are skipped. Invalid optional timestamps
or durations become `None` rather than failing the entire page.

The existing `CallLogService` performs the final selected-time-window
filtering against `started_at`. The raw payload remains available to the UI,
including status, participants, and recording links.

## Downloading Recordings

`VochiTelephonyAdapter.get_recording()` first requests:

`GET {base_url}/recording?unique_id={id}&key={api_key}`

The metadata response must contain a non-empty `download_url`. The adapter
then downloads that URL without the VoChi API key. It returns:

- downloaded bytes as `Recording.content`
- the download response content type when available, otherwise `audio/mpeg`
- permanent `recording_url` from metadata as `Recording.source_uri`

If `recording_url` is absent, `source_uri` falls back to the metadata endpoint
URL without its secret query parameter.

HTTP failures, malformed JSON, missing `calls`, missing `download_url`, and
recording-download failures are translated to `TelephonyError`. Error text
identifies the operation and call ID where relevant but never includes the
API key.

## UI Links

The UI must stop constructing legacy `/calllogs/{client_id}/{unique_id}`
links. It uses, in order:

1. `recording_url` from the call's raw API payload;
2. the source URI returned after a recording is downloaded;
3. `TenantConfig.recording_url(unique_id)` as a non-secret fallback.

This preserves permanent listen links while the one-hour S3 URL is used only
for downloading audio.

## Testing

Adapter tests cover:

- `/calls` requests with an explicitly empty `phone` parameter;
- pagination across calls for all phone numbers;
- exclusion of `call_status` values `0` and `1`;
- inclusion of answered calls with `call_status=2`;
- existing incoming, outgoing, and internal call-type filtering;
- multi-page pagination and termination;
- mapping valid and malformed call fields;
- skipping entries without `unique_id`;
- metadata lookup followed by S3 download;
- content type and permanent source URL handling;
- missing `download_url`, invalid payloads, and HTTP failures;
- absence of the API key from raised errors.

Tenant tests cover the new default base URL, normalization, required
`VOCHI_API_KEY`, and provider-specific recording URL generation.

UI handler tests cover selection of `recording_url` from raw call data instead
of legacy URL construction.

The full existing test suite must pass after the migration.
