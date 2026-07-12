from __future__ import annotations

from types import SimpleNamespace

from calls_analyser.ui import dependencies


class _FakeSecretsAdapter:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = values or {}

    def get_optional_secret(self, key: str) -> str | None:
        return self._values.get(key)


class _FakePromptService:
    def __init__(
        self,
        prompts,  # noqa: ANN001
        tenant_templates=None,  # noqa: ANN001
        *,
        prompt_repository=None,  # noqa: ANN001
    ) -> None:
        self.prompts = prompts
        self.tenant_templates = tenant_templates
        self.prompt_repository = prompt_repository


class _FakeSupabaseAuthRepository:
    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

    def get_user_by_login(self, _login: str):
        return None

    def get_user_by_id(self, _user_id: str):
        return None

    def get_tenant(self, _tenant_id: str):
        return None

    def list_access_for_user(self, _user_id: str) -> list:
        return []


class _FakeSupabaseTenantSettingsRepository:
    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

    def get_setting(self, _tenant_id: str, _key: str):
        return None

    def get_secret(self, _tenant_id: str, _key: str):
        return None

    def list_tenant_ids(self) -> list[str]:
        return []


class _FakeSupabasePromptTemplateRepository:
    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key


def _patch_build_dependencies(
    monkeypatch,  # noqa: ANN001
    batch_params: SimpleNamespace | None = None,
    secrets: dict[str, str] | None = None,
) -> None:
    monkeypatch.setattr(dependencies.config, "PROJECT_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(dependencies, "EnvSecretsAdapter", lambda: _FakeSecretsAdapter(secrets))
    monkeypatch.setattr(dependencies, "LocalStorageAdapter", lambda: "storage")
    monkeypatch.setattr(dependencies, "PromptService", _FakePromptService)
    monkeypatch.setattr(dependencies, "ProviderRegistry", lambda: {})
    monkeypatch.setattr(dependencies, "_register_gemini_models", lambda *_args: None)
    monkeypatch.setattr(dependencies, "_build_tenant_service", lambda *args, **kwargs: "tenant-service")
    monkeypatch.setattr(dependencies, "_build_call_log_service", lambda *_args: "call-log")
    monkeypatch.setattr(dependencies, "SupabaseCache", lambda url, key: ("cache", url, key))
    monkeypatch.setattr(dependencies, "SupabaseUsageTracker", lambda url, key: ("usage", url, key))
    monkeypatch.setattr(
        dependencies,
        "SupabaseUsageReportRepository",
        lambda url, key: ("usage-report", url, key),
    )
    monkeypatch.setattr(dependencies, "FileBackedCache", lambda path: ("file-cache", path))
    monkeypatch.setattr(dependencies, "AnalysisService", lambda *args, **kwargs: ("analysis", args, kwargs))
    monkeypatch.setattr(dependencies, "_build_email_report_service", lambda: None)
    monkeypatch.setattr(dependencies, "_build_model_options", lambda _registry: [])
    monkeypatch.setattr(dependencies, "load_batch_params", lambda: batch_params or SimpleNamespace())


def test_build_dependencies_wires_auth_service_from_env_user(monkeypatch) -> None:
    _patch_build_dependencies(monkeypatch)
    monkeypatch.setattr(dependencies.config, "DEFAULT_TENANT_ID", "tenant-default")
    monkeypatch.setenv("VOCHI_UI_LOGIN", "alice")
    monkeypatch.setenv("VOCHI_UI_PASSWORD", "correct-password")

    deps = dependencies.build_dependencies()

    user = deps.auth_service.authenticate("alice", "correct-password")
    assert user is not None
    assert user.login == "alice"
    assert user.roles_by_tenant == {"tenant-default": "admin"}
    assert deps.auth_service.authenticate("alice", "wrong-password") is None


def test_build_dependencies_wires_ui_batch_orchestration(monkeypatch) -> None:
    _patch_build_dependencies(monkeypatch)

    deps = dependencies.build_dependencies()

    assert isinstance(deps.sequential_batch_executor, dependencies.SequentialBatchExecutor)
    assert deps.sequential_batch_executor._analysis_service is deps.analysis_service
    assert isinstance(deps.batch_orchestrator, dependencies.BatchAnalysisOrchestrator)
    assert deps.batch_orchestrator._executor is deps.sequential_batch_executor
    assert deps.batch_orchestrator._prompt_service is deps.prompt_service
    assert deps.batch_orchestrator._ai_registry is deps.ai_registry


def test_build_dependencies_uses_admin_login_for_legacy_ui_password(monkeypatch) -> None:
    _patch_build_dependencies(monkeypatch)
    monkeypatch.setattr(dependencies.config, "DEFAULT_TENANT_ID", "tenant-default")
    monkeypatch.delenv("VOCHI_UI_LOGIN", raising=False)
    monkeypatch.setenv("VOCHI_UI_PASSWORD", "legacy-password")

    deps = dependencies.build_dependencies()

    user = deps.auth_service.authenticate("admin", "legacy-password")
    assert user is not None
    assert user.login == "admin"
    assert user.roles_by_tenant == {"tenant-default": "admin"}


def test_tenant_settings_service_resolves_batch_fallback_from_batch_params(monkeypatch) -> None:
    batch_params = SimpleNamespace(
        enable_gemini_batch=False,
        batch_size=37,
        scheduler_enabled=True,
        scheduler_mode="interval",
        scheduler_interval_minutes=45,
        filter_time_from="09:00",
        filter_time_to="18:00",
        filter_call_type="missed",
    )
    _patch_build_dependencies(monkeypatch, batch_params=batch_params)
    monkeypatch.setattr(dependencies.config, "DEFAULT_TENANT_ID", "tenant-default")
    monkeypatch.delenv("VOCHI_UI_PASSWORD", raising=False)

    deps = dependencies.build_dependencies()

    settings = deps.tenant_settings_service.resolve("tenant-default")
    assert settings.batch_enabled is False
    assert settings.batch_size == 37
    assert settings.scheduler_enabled is True
    assert settings.scheduler_mode == "interval"
    assert settings.scheduler_interval_minutes == 45
    assert settings.scheduler_filters == {
        "time_from": "09:00",
        "time_to": "18:00",
        "call_type": "missed",
    }


def test_build_dependencies_uses_supabase_repositories_when_configured(monkeypatch) -> None:
    _patch_build_dependencies(
        monkeypatch,
        secrets={
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "service-key",
        },
    )
    monkeypatch.setattr(dependencies, "SupabaseAuthRepository", _FakeSupabaseAuthRepository)
    monkeypatch.setattr(
        dependencies,
        "SupabaseTenantSettingsRepository",
        _FakeSupabaseTenantSettingsRepository,
    )
    monkeypatch.setattr(
        dependencies,
        "SupabasePromptTemplateRepository",
        _FakeSupabasePromptTemplateRepository,
    )

    deps = dependencies.build_dependencies()

    assert isinstance(deps.auth_service._repository, _FakeSupabaseAuthRepository)
    assert isinstance(
        deps.tenant_settings_service._repository,
        _FakeSupabaseTenantSettingsRepository,
    )
    assert isinstance(deps.prompt_service.prompt_repository, _FakeSupabasePromptTemplateRepository)
    assert deps.auth_service._repository.supabase_key == "service-key"


def test_build_dependencies_falls_back_when_supabase_repository_wiring_fails(monkeypatch) -> None:
    class _BrokenRepository:
        def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise RuntimeError("supabase unavailable")

    _patch_build_dependencies(
        monkeypatch,
        secrets={
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "service-key",
        },
    )
    monkeypatch.setattr(dependencies.config, "DEFAULT_TENANT_ID", "tenant-default")
    monkeypatch.setenv("VOCHI_UI_LOGIN", "alice")
    monkeypatch.setenv("VOCHI_UI_PASSWORD", "correct-password")
    monkeypatch.setattr(dependencies, "SupabaseAuthRepository", _BrokenRepository)
    monkeypatch.setattr(dependencies, "SupabaseTenantSettingsRepository", _BrokenRepository)
    monkeypatch.setattr(dependencies, "SupabasePromptTemplateRepository", _BrokenRepository)

    deps = dependencies.build_dependencies()

    assert deps.auth_service.authenticate("alice", "correct-password") is not None
    assert deps.tenant_settings_service.resolve("tenant-default").batch_size == 20
    assert deps.prompt_service.prompt_repository is None


def test_build_dependencies_wires_supabase_tenant_source_into_tenant_service(monkeypatch) -> None:
    captured: dict[str, object] = {}

    _patch_build_dependencies(
        monkeypatch,
        secrets={
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "service-key",
        },
    )
    monkeypatch.setattr(dependencies, "SupabaseAuthRepository", _FakeSupabaseAuthRepository)
    monkeypatch.setattr(
        dependencies,
        "SupabaseTenantSettingsRepository",
        _FakeSupabaseTenantSettingsRepository,
    )
    monkeypatch.setattr(
        dependencies,
        "SupabasePromptTemplateRepository",
        _FakeSupabasePromptTemplateRepository,
    )
    def _capture_tenant_service(
        secrets_adapter,  # noqa: ANN001
        *,
        tenant_settings_source=None,  # noqa: ANN001
    ) -> str:
        captured["secrets_adapter"] = secrets_adapter
        captured["tenant_settings_source"] = tenant_settings_source
        return "tenant-service"

    monkeypatch.setattr(dependencies, "_build_tenant_service", _capture_tenant_service)
    monkeypatch.setattr(dependencies.config, "DEFAULT_TENANT_ID", "tenant-default")
    monkeypatch.setattr(dependencies.config, "DEFAULT_BASE_URL", "https://default.example/api/v1")

    deps = dependencies.build_dependencies()

    assert deps.tenant_service == "tenant-service"
    assert isinstance(
        captured["tenant_settings_source"],
        _FakeSupabaseTenantSettingsRepository,
    )
