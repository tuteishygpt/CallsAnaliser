# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calls Analyser is a Python application that analyzes telephony call recordings using Google Gemini AI. It follows hexagonal (ports & adapters) architecture. Two telephony providers are supported: Vochi CRM and MTS VATS. Results are cached in Supabase or local file storage. A Gradio web UI and CLI batch runner are the main entry points.

## Commands

```bash
# Run the Gradio UI (launches on localhost:7860)
python app.py

# Run all tests
pytest

# Run a single test
pytest tests/test_analysis_service.py::test_analysis_service_is_idempotent -v

# Run CLI batch for yesterday's calls
python -m calls_analyser.runner

# Run CLI batch for a specific date
python -m calls_analyser.runner --date 2024-04-10 --time-from 09:00 --time-to 18:00 --tenant-id Multicom

# Install with dev dependencies
pip install -e ".[dev]"
```

## Architecture

### Hexagonal layers

- **Domain** (`calls_analyser/domain/`): Pydantic models (`CallLogEntry`, `Recording`, `RecordingHandle`, `AnalysisResult`, `Language` enum) and exception hierarchy. No external dependencies.
- **Ports** (`calls_analyser/ports/`): Abstract interfaces — `TelephonyPort`, `AIModelPort`, `StoragePort`, `SecretsPort`. All adapters implement these.
- **Adapters** (`calls_analyser/adapters/`): Concrete implementations — `VochiTelephonyAdapter`, `MtsVatsTelephonyAdapter`, `GeminiAIAdapter`, `LocalStorageAdapter`, `SupabaseCache`, `EnvSecretsAdapter`.
- **Services** (`calls_analyser/services/`): Business logic — `AnalysisService` (orchestrates analysis with idempotent caching), `CallLogService`, `TenantService`, `PromptService`, `ProviderRegistry`, `GeminiBatchRunner`, `FileBackedCache`.
- **UI** (`calls_analyser/ui/`): Gradio-based web interface. `dependencies.py` contains `AppDependencies` dataclass and `build_dependencies()` which wires all adapters and services together.

### Key data flow

1. `AnalysisService.analyze_call()` checks cache by composite key `(tenant_id, call_id, prompt_key, provider_name, model_key, custom_prompt)`
2. On cache miss: `CallLogService.ensure_recording()` downloads audio via telephony adapter, then `AIModelPort.analyze()` processes it
3. Result is cached and returned. Same inputs always return the cached result (idempotency).

### Multi-tenant resolution

`TenantService.resolve()` reads env vars prefixed by tenant ID (e.g., `MULTICOM_TELEPHONY_PROVIDER`, `MULTICOM_MTS_API_KEY`) to build `TenantConfig`. The `DEFAULT_TENANT_ID` env var sets the fallback tenant.

### Entry points

- `app.py` — Gradio UI + APScheduler background jobs
- `calls_analyser/runner.py` — CLI batch runner (`python -m calls_analyser.runner`)
- `calls_analyser/api/http.py` — FastAPI HTTP layer

## Configuration

- `.env` — Secrets and tenant config (git-ignored). Required keys: `GOOGLE_API_KEY`, tenant-specific vars (`<TENANT>_TELEPHONY_PROVIDER`, `<TENANT>_MTS_DOMAIN`, `<TENANT>_MTS_API_KEY` for MTS VATS, or `<TENANT>_VOCHI_API_KEY` for VoChi API v1).
- `batch_params.json` — Batch processing and scheduler settings (`enable_gemini_batch`, `batch_size`, scheduler cron/interval config, filters).
- Optional: `SUPABASE_URL` + `SUPABASE_KEY` env vars enable Supabase cache; without them, `FileBackedCache` is used.

## Testing patterns

Tests use pytest with monkeypatch-based dependency injection. Adapter stubs/fakes (e.g., `FakeAIModel`, `StubCallLogService`) are defined inline in test files. `conftest.py` adds project root to `sys.path`. Tests in `test_app_batch.py` monkeypatch module-level globals in `app.py` and call `ui_mass_analyze()`.
