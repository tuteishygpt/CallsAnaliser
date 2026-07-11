import base64
import json
import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clear_credentials_cache():
    from calls_analyser import google_credentials

    google_credentials.load_google_credentials.cache_clear()
    yield
    google_credentials.load_google_credentials.cache_clear()


def _encoded_service_account(marker: str = "private-secret-marker") -> str:
    payload = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "key-id",
        "private_key": marker,
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def test_valid_b64_credentials_are_built_with_cloud_platform_scope_and_cached(monkeypatch):
    from calls_analyser import google_credentials

    encoded = _encoded_service_account()
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", encoded)
    calls = []
    credential = object()

    def from_info(info, *, scopes=None):
        calls.append((info, scopes))
        return credential

    monkeypatch.setattr(
        google_credentials.service_account.Credentials,
        "from_service_account_info",
        from_info,
    )

    assert google_credentials.load_google_credentials() is credential
    assert google_credentials.load_google_credentials() is credential
    assert calls == [
        (
            json.loads(base64.b64decode(encoded)),
            ["https://www.googleapis.com/auth/cloud-platform"],
        )
    ]


def test_malformed_b64_returns_none_without_logging_secret(monkeypatch, caplog):
    from calls_analyser import google_credentials

    secret = "not-valid-base64-secret"
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", secret)

    with caplog.at_level(logging.WARNING):
        assert google_credentials.load_google_credentials() is None

    assert secret not in caplog.text
    assert "private_key" not in caplog.text


def test_gemini_api_key_takes_precedence_over_malformed_b64(monkeypatch):
    from calls_analyser.adapters.ai import gemini

    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "malformed")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    seen = []

    adapter = gemini.GeminiAIAdapter(
        api_key="api-key",
        model="models/gemini-test",
        client_factory=lambda key: seen.append(key) or object(),
    )

    assert adapter._client is not None
    assert seen == ["api-key"]


def test_gemini_api_key_does_not_construct_valid_b64_credentials(monkeypatch):
    from calls_analyser.adapters.ai import gemini

    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", _encoded_service_account())
    monkeypatch.setattr(
        gemini,
        "load_google_credentials",
        lambda: pytest.fail("service-account credentials must not be loaded for an API key"),
    )

    gemini.GeminiAIAdapter(
        api_key="api-key",
        model="models/gemini-test",
        client_factory=lambda key: object(),
    )


@pytest.mark.parametrize(
    "decoded",
    ["not-json", json.dumps(["not", "an", "object"])],
)
def test_decoded_invalid_json_falls_back_with_sanitized_log(
    monkeypatch, caplog, decoded
):
    from calls_analyser import google_credentials

    encoded = base64.b64encode(decoded.encode()).decode()
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", encoded)

    with caplog.at_level(logging.WARNING):
        assert google_credentials.load_google_credentials() is None

    assert encoded not in caplog.text
    assert decoded not in caplog.text
    assert "Expecting value" not in caplog.text


def test_rejected_service_account_info_falls_back_without_exception_leakage(
    monkeypatch, caplog
):
    from calls_analyser import google_credentials

    marker = "private-secret-marker"
    exception_marker = "credential-library-internal-detail"
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", _encoded_service_account(marker))
    monkeypatch.setattr(
        google_credentials.service_account.Credentials,
        "from_service_account_info",
        lambda info: (_ for _ in ()).throw(ValueError(exception_marker)),
    )

    with caplog.at_level(logging.WARNING):
        assert google_credentials.load_google_credentials() is None

    assert marker not in caplog.text
    assert exception_marker not in caplog.text


def test_gemini_default_factory_uses_explicit_b64_credentials(monkeypatch):
    from calls_analyser.adapters.ai import gemini

    credential = object()
    monkeypatch.setattr(gemini, "load_google_credentials", lambda: credential)
    calls = []
    monkeypatch.setattr(gemini.genai, "Client", lambda **kwargs: calls.append(kwargs) or object())

    gemini.GeminiAIAdapter(
        api_key=None, model="models/gemini-test", project="test-project"
    )

    assert calls == [{
        "vertexai": True,
        "project": "test-project",
        "location": "global",
        "credentials": credential,
    }]


def test_malformed_b64_does_not_block_adc(monkeypatch):
    from calls_analyser.adapters.ai import gemini

    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "malformed")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "existing-adc.json")
    monkeypatch.setattr(gemini, "load_google_credentials", lambda: None)
    calls = []
    monkeypatch.setattr(gemini.genai, "Client", lambda **kwargs: calls.append(kwargs) or object())

    gemini.GeminiAIAdapter(
        api_key=None, model="models/gemini-test", project="test-project"
    )

    assert calls == [{
        "vertexai": True,
        "project": "test-project",
        "location": "global",
    }]


def test_batch_clients_share_explicit_b64_credentials(monkeypatch):
    from calls_analyser.services import gemini_batch

    credential = object()
    monkeypatch.setattr(gemini_batch, "load_google_credentials", lambda: credential)
    genai_calls = []
    storage_calls = []
    monkeypatch.setattr(
        gemini_batch.genai,
        "Client",
        lambda **kwargs: genai_calls.append(kwargs) or object(),
    )

    class StorageClient:
        def __init__(self, **kwargs):
            storage_calls.append(kwargs)

        def bucket(self, name):
            return name

    monkeypatch.setattr(gemini_batch.gcs_storage, "Client", StorageClient)

    gemini_batch.VertexBatchRunner(model="models/gemini-test", bucket="bucket")

    assert genai_calls[0]["credentials"] is credential
    assert storage_calls[0]["credentials"] is credential


def test_batch_adc_mode_omits_explicit_credentials(monkeypatch):
    from calls_analyser.services import gemini_batch

    monkeypatch.setattr(gemini_batch, "load_google_credentials", lambda: None)
    genai_calls = []
    storage_calls = []
    monkeypatch.setattr(
        gemini_batch.genai,
        "Client",
        lambda **kwargs: genai_calls.append(kwargs) or object(),
    )

    class StorageClient:
        def __init__(self, **kwargs):
            storage_calls.append(kwargs)

        def bucket(self, name):
            return name

    monkeypatch.setattr(gemini_batch.gcs_storage, "Client", StorageClient)

    gemini_batch.VertexBatchRunner(model="models/gemini-test", bucket="bucket")

    assert "credentials" not in genai_calls[0]
    assert "credentials" not in storage_calls[0]


def test_ui_registers_models_with_b64_credentials_only(monkeypatch):
    from calls_analyser.ui import dependencies
    from calls_analyser.services.registry import ProviderRegistry

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(dependencies, "load_google_credentials", lambda: object())

    class Secrets:
        def get_optional_secret(self, name):
            return None

    class Adapter:
        provider_name = "gemini"

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(dependencies, "GeminiAIAdapter", Adapter)
    registry = ProviderRegistry()

    dependencies._register_gemini_models(registry, Secrets())

    assert len(registry) == len(dependencies.config.MODEL_CANDIDATES)


def test_model_candidates_include_current_gemini_audio_analysis_models():
    from calls_analyser import config

    model_keys = {model_key for _title, model_key in config.MODEL_CANDIDATES}

    assert {
        "models/gemini-3.5-flash",
        "models/gemini-3.1-flash-lite",
        "models/gemini-3.1-pro-preview",
        "models/gemini-3-flash-preview",
    } <= model_keys


def test_entrypoints_have_no_credential_tempfile_bootstrap():
    root = Path(__file__).parents[1]
    for relative in ("app.py", "calls_analyser/runner.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "GOOGLE_SERVICE_ACCOUNT_JSON_B64" not in source
        assert "tempfile" not in source


def test_batch_runner_documentation_describes_in_memory_b64_credentials():
    from calls_analyser.services.gemini_batch import VertexBatchRunner

    doc = VertexBatchRunner.__doc__ or ""
    assert "GOOGLE_SERVICE_ACCOUNT_JSON_B64" in doc
    assert "write to a temp file" not in doc
