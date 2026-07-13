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


class _FakeTenantRepository:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        *,
        codec=None,  # noqa: ANN001
    ) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.codec = codec

    def get_setting(self, _tenant_id: str, _key: str):
        return None

    def get_secret(self, _tenant_id: str, _key: str):
        return None

    def list_tenant_ids(self) -> list[str]:
        return []


class _TenantServiceSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> str:
        self.calls.append({"args": args, "kwargs": kwargs})
        return "tenant-service"


def _patch_build_dependencies(monkeypatch, secrets: dict[str, str]) -> None:  # noqa: ANN001
    monkeypatch.setattr(dependencies.config, "PROJECT_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(dependencies.config, "Language", lambda value: value)
    monkeypatch.setattr(dependencies, "EnvSecretsAdapter", lambda: _FakeSecretsAdapter(secrets))
    monkeypatch.setattr(dependencies, "LocalStorageAdapter", lambda: "storage")
    monkeypatch.setattr(
        dependencies,
        "PromptService",
        lambda prompts, *, prompt_repository=None: (
            "prompts",
            prompts,
            prompt_repository,
        ),
    )
    monkeypatch.setattr(dependencies, "ProviderRegistry", lambda: {})
    monkeypatch.setattr(dependencies, "_register_gemini_models", lambda *_args: None)
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
    tenant_service_spy = _TenantServiceSpy()
    monkeypatch.setattr(dependencies, "_build_tenant_service", tenant_service_spy)
    _patch_build_dependencies(
        monkeypatch,
        {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "service-key",
        },
    )
    monkeypatch.setattr(
        dependencies,
        "SupabaseTenantSettingsRepository",
        _FakeTenantRepository,
    )

    deps = dependencies.build_dependencies()

    assert isinstance(deps.usage_report_repository, _FakeReportRepository)
    assert deps.usage_report_repository.supabase_url == "https://example.supabase.co"
    assert deps.usage_report_repository.supabase_key == "service-key"
    repository = deps.tenant_settings_service._repository
    assert isinstance(repository, _FakeTenantRepository)
    assert tenant_service_spy.calls[0]["kwargs"]["tenant_settings_source"] is repository
    assert deps.tenant_admin_settings_service._repository is repository
    assert deps.prompt_service[2] is repository


def test_build_dependencies_leaves_usage_report_repository_empty_without_supabase(
    monkeypatch,
) -> None:
    tenant_service_spy = _TenantServiceSpy()
    monkeypatch.setattr(dependencies, "_build_tenant_service", tenant_service_spy)
    _patch_build_dependencies(monkeypatch, {})

    deps = dependencies.build_dependencies()

    assert deps.usage_report_repository is None
    assert tenant_service_spy.calls[0]["kwargs"]["tenant_settings_source"] is not None
