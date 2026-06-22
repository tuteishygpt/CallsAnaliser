# VoChi API v1 Integration Design

## Goal

Replace the legacy `crm.vochi.by/api/calllogs` integration with the new
`bot.vochi.by/api/v1` API. The application must list all unsuccessful
external calls for a selected day and download available recordings for
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

`GET {base_url}/unsuccessful-calls`

with these query parameters:

- `key`: configured API key
- `date_from`: selected day in `YYYY-MM-DD`
- `date_to`: the same selected day
- `direction`: derived from the existing call-type filter:
  - no filter -> `all`
  - incoming (`0`) -> `incoming`
  - outgoing (`1`) -> `outgoing`
  - internal (`2`) -> return an empty result without an HTTP request because
    the endpoint excludes internal calls
- `limit`: `100`
- `offset`: starts at `0` and increases by the number of returned calls

The adapter follows pagination until one of these conditions is met:

- the number of collected calls reaches `total`;
- a page contains fewer than `limit` calls;
- a page is empty.

This prevents an infinite loop if the server returns inconsistent pagination
metadata.

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

- request parameters for all/incoming/outgoing directions;
- internal-call filtering without an HTTP request;
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
