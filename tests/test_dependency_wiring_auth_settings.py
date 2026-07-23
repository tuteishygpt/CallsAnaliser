from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from calls_analyser.services.prompt import PromptService, PromptTemplate
from calls_analyser.services.tenant import TenantService
from calls_analyser.services.tenant_admin_settings import InMemoryTenantAdminRepository
from calls_analyser.services.tenant_secret_codec import TenantSecretCodec
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
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        *,
        codec: TenantSecretCodec | None = None,
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
    monkeypatch.setattr(dependencies.config, "Language", lambda value: value)
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
        "SupabaseSchedulerRunRepository",
        lambda url, key: ("scheduler-runs", url, key),
    )
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
    assert deps.prompt_service.prompt_repository is deps.tenant_settings_service._repository
    assert deps.tenant_admin_settings_service._repository is deps.tenant_settings_service._repository
    assert isinstance(deps.tenant_settings_service._repository.codec, TenantSecretCodec)
    assert deps.auth_service._repository.supabase_key == "service-key"
    assert deps.scheduler_run_repository == (
        "scheduler-runs",
        "https://example.supabase.co",
        "service-key",
    )


@pytest.mark.parametrize(
    "secrets",
    [
        {},
        {"SUPABASE_URL": "https://example.supabase.co"},
        {"SUPABASE_KEY": "service-key"},
    ],
)
def test_build_dependencies_leaves_scheduler_repository_unwired_without_complete_credentials(
    monkeypatch,
    secrets,
) -> None:
    _patch_build_dependencies(monkeypatch, secrets=secrets)

    deps = dependencies.build_dependencies()

    assert deps.scheduler_run_repository is None


@pytest.mark.parametrize(
    ("supabase_url", "supabase_key"),
    [
        ("   ", "service-key"),
        ("https://example.supabase.co", "   "),
    ],
)
def test_scheduler_repository_builder_rejects_whitespace_credentials(
    monkeypatch,
    supabase_url,
    supabase_key,
) -> None:
    constructor_calls = []
    monkeypatch.setattr(
        dependencies,
        "SupabaseSchedulerRunRepository",
        lambda *args: constructor_calls.append(args) or object(),
    )

    repository = dependencies._build_scheduler_run_repository(
        supabase_url,
        supabase_key,
    )

    assert repository is None
    assert constructor_calls == []


def test_build_dependencies_leaves_scheduler_repository_unwired_when_constructor_fails(
    monkeypatch,
) -> None:
    class BrokenSchedulerRepository:
        def __init__(self, *_args) -> None:
            raise RuntimeError("scheduler_runs unavailable")

    _patch_build_dependencies(
        monkeypatch,
        secrets={
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_KEY": "service-key",
        },
    )
    monkeypatch.setattr(
        dependencies,
        "SupabaseSchedulerRunRepository",
        BrokenSchedulerRepository,
    )

    deps = dependencies.build_dependencies()

    assert deps.scheduler_run_repository is None


def test_tenant_repository_builder_passes_codec_through_public_constructor(monkeypatch) -> None:
    monkeypatch.setattr(
        dependencies,
        "SupabaseTenantSettingsRepository",
        _FakeSupabaseTenantSettingsRepository,
    )
    codec = TenantSecretCodec(
        base64.urlsafe_b64encode(b"k" * 32).decode("ascii").rstrip("=")
    )

    repository = dependencies._build_tenant_settings_repository(
        "https://example.supabase.co",
        "service-key",
        codec=codec,
    )

    assert repository.codec is codec


def test_build_dependencies_disables_admin_when_supabase_repository_wiring_fails(
    monkeypatch,
) -> None:
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
    in_memory_admin_constructions: list[dict[str, object]] = []
    original_in_memory_repository = InMemoryTenantAdminRepository

    def _track_in_memory_admin(**kwargs):  # noqa: ANN003, ANN202
        in_memory_admin_constructions.append(kwargs)
        return original_in_memory_repository(**kwargs)

    monkeypatch.setattr(dependencies, "InMemoryTenantAdminRepository", _track_in_memory_admin)
    tenant_sources: list[object | None] = []
    monkeypatch.setattr(
        dependencies,
        "_build_tenant_service",
        lambda _secrets, *, tenant_settings_source=None: tenant_sources.append(
            tenant_settings_source
        )
        or "tenant-service",
    )

    deps = dependencies.build_dependencies()

    assert deps.auth_service.authenticate("alice", "correct-password") is not None
    assert deps.tenant_settings_service.resolve("tenant-default").batch_size == 20
    assert deps.prompt_service.prompt_repository is None
    assert deps.tenant_admin_settings_service is None
    assert tenant_sources == [None]
    assert in_memory_admin_constructions == []


def test_tenant_settings_builder_preserves_a_falsey_repository(monkeypatch) -> None:
    class _FalseyRepository:
        def __bool__(self) -> bool:
            return False

        def get_setting(self, _tenant_id: str, _key: str):
            return None

        def get_secret(self, _tenant_id: str, _key: str):
            return None

        def list_tenant_ids(self) -> list[str]:
            return []

    repository = _FalseyRepository()
    monkeypatch.setattr(
        dependencies,
        "_build_tenant_settings_repository",
        lambda *_args, **_kwargs: pytest.fail("repository must not be rebuilt"),
    )

    service = dependencies._build_tenant_settings_service(
        SimpleNamespace(),
        repository=repository,
    )

    assert service._repository is repository


def test_prompt_service_does_not_hide_repository_constructor_type_errors(monkeypatch) -> None:
    class _PromptServiceWithBrokenRepositoryConstructor:
        def __init__(self, _prompts, *, prompt_repository=None) -> None:  # noqa: ANN001
            if prompt_repository is not None:
                raise TypeError("repository-aware constructor failed")

    monkeypatch.setattr(
        dependencies,
        "PromptService",
        _PromptServiceWithBrokenRepositoryConstructor,
    )

    with pytest.raises(TypeError, match="repository-aware constructor failed"):
        dependencies._build_prompt_service(repository=object())


def test_build_dependencies_uses_minimal_fallback_when_admin_imports_are_missing(
    monkeypatch,
) -> None:
    _patch_build_dependencies(monkeypatch)
    monkeypatch.setattr(dependencies, "TenantSecretCodec", None)
    monkeypatch.setattr(dependencies, "InMemoryTenantAdminRepository", None)
    monkeypatch.setattr(dependencies, "TenantAdminSettingsService", None)

    deps = dependencies.build_dependencies()

    assert deps.project_imports_available is False
    assert deps.tenant_admin_settings_service is None


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


def test_local_admin_save_is_immediately_visible_to_all_runtime_services(monkeypatch) -> None:
    master_key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii").rstrip("=")
    repository = InMemoryTenantAdminRepository(
        tenants={"tenant-default": {"display_name": "Tenant", "status": "active"}},
        settings={},
        secrets={},
        prompts={
            "tenant-default": [
                {
                    "key": "tenant-analysis",
                    "title": "Tenant analysis",
                    "body": "Read-only tenant prompt",
                    "version": 1,
                    "is_active": True,
                }
            ]
        },
        codec=TenantSecretCodec(master_key),
    )

    _patch_build_dependencies(
        monkeypatch,
        secrets={"TENANT_SECRETS_MASTER_KEY": master_key},
    )
    monkeypatch.setattr(dependencies.config, "DEFAULT_TENANT_ID", "tenant-default")
    monkeypatch.setattr(dependencies, "PromptService", PromptService)
    monkeypatch.setattr(dependencies, "InMemoryTenantAdminRepository", lambda **_kwargs: repository)
    monkeypatch.setattr(
        dependencies,
        "_build_tenant_service",
        lambda secrets_adapter, *, tenant_settings_source=None: TenantService(
            secrets_adapter,
            default_tenant="tenant-default",
            tenant_settings_source=tenant_settings_source,
        ),
    )

    deps = dependencies.build_dependencies()
    document = deps.tenant_admin_settings_service.load("tenant-default")
    document.update(
        telephony_provider="vochi",
        vochi_base_url="https://tenant.example/api/v1",
        vochi_api_key="tenant-api-key",
        default_language="be",
        batch_size=41,
    )

    deps.tenant_admin_settings_service.save("tenant-default", document, "admin-user")

    assert deps.tenant_admin_settings_service._repository is repository
    assert deps.tenant_settings_service._repository is repository
    assert deps.tenant_service._tenant_settings_source is repository
    assert deps.prompt_service._prompt_repository is repository
    runtime_settings = deps.tenant_settings_service.resolve("tenant-default")
    assert runtime_settings.default_language == "be"
    assert runtime_settings.batch_size == 41
    telephony = deps.tenant_service.resolve("tenant-default")
    assert telephony.vochi_base_url == "https://tenant.example/api/v1"
    assert telephony.vochi_api_key == "tenant-api-key"
    assert deps.prompt_service.get_prompt(
        "tenant-analysis",
        tenant_id="tenant-default",
    ) == PromptTemplate(
        key="tenant-analysis",
        title="Tenant analysis",
        body="Read-only tenant prompt",
        version=1,
    )
