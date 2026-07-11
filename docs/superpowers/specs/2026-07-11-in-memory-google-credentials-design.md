# In-memory Google credentials design

## Goal

Use the existing `GOOGLE_SERVICE_ACCOUNT_JSON_B64` Hugging Face secret without writing service-account JSON to the container filesystem.

## Design

Add a focused credentials loader that decodes and validates the secret and returns a `google.oauth2.service_account.Credentials` object. The loader caches the object for the process lifetime. Gemini and GCS clients receive that exact object explicitly.

Credential precedence is: `GOOGLE_API_KEY`, then valid `GOOGLE_SERVICE_ACCOUNT_JSON_B64`, then local ADC through `GOOGLE_APPLICATION_CREDENTIALS`. Malformed base64 or JSON never blocks a valid API key or ADC configuration.

Remove credential-file bootstrap code from `app.py` and `calls_analyser/runner.py`. `GOOGLE_APPLICATION_CREDENTIALS` remains supported for local ADC environments, but the base64 secret path never creates a file.

## Error handling

Invalid base64 or invalid service-account JSON is treated as unavailable credentials and logged through the module logger using a sanitized category-only message. Logs never include decoded bytes, JSON, exception text, or credential fields. Existing API-key and ADC paths continue to work.

## Verification

Tests must prove that credentials are decoded once, reused, passed to both Gemini and GCS clients, and omitted in ADC mode. Static regression checks must prove credential bootstrap and credential-related `tempfile` usage are absent from `app.py` and `runner.py`. UI model registration and help text must recognize B64-only configuration.

Deployment verification must confirm the Space is running, `GOOGLE_APPLICATION_CREDENTIALS` is unset, startup logs contain no credential-file message, and an authenticated Google client path initializes from the B64-only configuration without a filesystem credential.
