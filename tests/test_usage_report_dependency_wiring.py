from __future__ import annotations

from types import SimpleNamespace

from calls_analyser.ui import dependencies


class _FakeSecretsAdapter:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get_optional_secret(self, key: str) -> str | None:
        return self._values.get(key)


class _FakeReportRepository:
    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key


def _patch_build_dependencies(monkeypatch, secrets: dict[str, str]) -> None:  # noqa: ANN001
    monkeypatch.setattr(dependencies.config, "PROJECT_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(dependencies, "EnvSecretsAdapter", lambda: _FakeSecretsAdapter(secrets))
    monkeypatch.setattr(dependencies, "LocalStorageAdapter", lambda: "storage")
    monkeypatch.setattr(dependencies, "PromptService", lambda prompts: ("prompts", prompts))
    monkeypatch.setattr(dependencies, "ProviderRegistry", lambda: {})
    monkeypatch.setattr(dependencies, "_register_gemini_models", lambda *_args: None)
    monkeypatch.setattr(dependencies, "_build_tenant_service", lambda *_args: "tenant-service")
    monkeypatch.setattr(dependencies, "_build_call_log_service", lambda *_args: "call-log")
    monkeypatch.setattr(dependencies, "SupabaseCache", lambda url, key: ("cache", url, key))
    monkeypatch.setattr(dependencies, "SupabaseUsageTracker", lambda url, key: ("usage", url, key))
    monkeypatch.setattr(dependencies, "SupabaseUsageReportRepository", _FakeReportRepository, raising=False)
    monkeypatch.setattr(dependencies, "FileBackedCache", lambda path: ("file-cache", path))
    monkeypatch.setattr(dependencies, "AnalysisService", lambda *args, **kwargs: ("analysis", args, kwargs))
    monkeypatch.setattr(dependencies, "_build_email_report_service", lambda: None)
    monkeypatch.setattr(dependencies, "_build_model_options", lambda _registry: [])
    monkeypatch.setattr(dependencies, "load_batch_params", lambda: SimpleNamespace())


def test_build_dependencies_wires_usage_report_repository_when_supabase_configured(
    monkeypatch,
) -> None:
    _patch_build_dependencies(
        monkeypatch,
        {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "service-key",
        },
    )

    deps = dependencies.build_dependencies()

    assert isinstance(deps.usage_report_repository, _FakeReportRepository)
    assert deps.usage_report_repository.supabase_url == "https://example.supabase.co"
    assert deps.usage_report_repository.supabase_key == "service-key"


def test_build_dependencies_leaves_usage_report_repository_empty_without_supabase(
    monkeypatch,
) -> None:
    _patch_build_dependencies(monkeypatch, {})

    deps = dependencies.build_dependencies()

    assert deps.usage_report_repository is None
