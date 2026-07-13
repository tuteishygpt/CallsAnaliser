"""Dependency wiring for the Gradio UI."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

from . import config
from calls_analyser.batch_params import BatchParams, load_batch_params
from calls_analyser.google_credentials import load_google_credentials

try:  # pragma: no cover - optional imports
    from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
    from calls_analyser.adapters.mail.brevo import BrevoHTTPSAdapter
    from calls_analyser.adapters.mail.gmail import GMAIL_ADDRESS, GmailSMTPAdapter
    from calls_analyser.adapters.secrets.env import EnvSecretsAdapter
    from calls_analyser.adapters.storage.local import LocalStorageAdapter
    from calls_analyser.adapters.storage.supabase_storage import SupabaseCache
    from calls_analyser.adapters.storage.supabase_tenant import (
        SupabaseAuthRepository,
        SupabasePromptTemplateRepository,
        SupabaseTenantSettingsRepository,
    )
    from calls_analyser.adapters.storage.supabase_usage import SupabaseUsageTracker
    from calls_analyser.adapters.storage.supabase_usage_report import SupabaseUsageReportRepository
    from calls_analyser.adapters.telephony.mts_vats import MtsVatsTelephonyAdapter
    from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter
    from calls_analyser.domain.exceptions import CallsAnalyserError
    from calls_analyser.ports.ai import AIModelPort
    from calls_analyser.services.analysis import AnalysisOptions, AnalysisService
    from calls_analyser.services.cache import FileBackedCache
    from calls_analyser.services.call_log import CallLogService
    from calls_analyser.services.email_report import EmailReportService
    from calls_analyser.services.prompt import PromptService
    from calls_analyser.services.registry import ProviderRegistry
    from calls_analyser.services.telephony_factory import default_telephony_provider_factory
    from calls_analyser.services.tenant import TenantService
except ImportError:  # pragma: no cover - executed when project deps unavailable
    GeminiAIAdapter = None  # type: ignore
    BrevoHTTPSAdapter = None  # type: ignore
    GMAIL_ADDRESS = "tuttstt@gmail.com"
    GmailSMTPAdapter = None  # type: ignore
    EnvSecretsAdapter = None  # type: ignore
    LocalStorageAdapter = None  # type: ignore
    SupabaseCache = None  # type: ignore
    SupabaseAuthRepository = None  # type: ignore
    SupabasePromptTemplateRepository = None  # type: ignore
    SupabaseTenantSettingsRepository = None  # type: ignore
    SupabaseUsageTracker = None  # type: ignore
    SupabaseUsageReportRepository = None  # type: ignore
    MtsVatsTelephonyAdapter = None  # type: ignore
    VochiTelephonyAdapter = None  # type: ignore
    CallsAnalyserError = Exception  # type: ignore
    AIModelPort = Any  # type: ignore
    AnalysisOptions = None  # type: ignore
    AnalysisOptions = None  # type: ignore
    AnalysisService = None  # type: ignore
    FileBackedCache = None  # type: ignore
    CallLogService = None  # type: ignore
    EmailReportService = None  # type: ignore
    PromptService = None  # type: ignore
    ProviderRegistry = Dict  # type: ignore
    default_telephony_provider_factory = None  # type: ignore
    TenantService = None  # type: ignore

try:
    from calls_analyser.services.auth import AuthService, InMemoryAuthRepository, hash_password
    from calls_analyser.services.tenant_admin_settings import (
        InMemoryTenantAdminRepository,
        TenantAdminSettingsService,
    )
    from calls_analyser.services.tenant_secret_codec import TenantSecretCodec
    from calls_analyser.services.tenant_settings import (
        InMemoryTenantSettingsRepository,
        TenantSettingsService,
    )
except ImportError:  # pragma: no cover - service modules are part of the project
    AuthService = None  # type: ignore
    InMemoryAuthRepository = None  # type: ignore
    hash_password = None  # type: ignore
    InMemoryTenantAdminRepository = None  # type: ignore
    TenantAdminSettingsService = None  # type: ignore
    TenantSecretCodec = None  # type: ignore
    InMemoryTenantSettingsRepository = None  # type: ignore
    TenantSettingsService = None  # type: ignore


@dataclass
class AppDependencies:
    project_imports_available: bool
    secrets_adapter: Any
    storage_adapter: Any
    prompt_service: Any
    ai_registry: Any
    tenant_service: Any
    call_log_service: Any
    analysis_service: Any
    usage_tracker: Any
    usage_report_repository: Any
    email_report_service: Any
    model_options: List[Tuple[str, str]]
    model_choices: List[Tuple[str, str]]
    model_default: str
    model_info: str
    batch_prompt_key: str
    batch_prompt_text: str
    batch_model_key: str
    batch_language: config.Language
    batch_custom_conditions: str
    batch_custom_prompt_template: str
    batch_params: BatchParams
    auth_service: Any = None
    tenant_settings_service: Any = None
    tenant_admin_settings_service: Any = None


MODEL_PLACEHOLDER_CHOICE = (
    "Configure Vertex AI credentials to enable Gemini models",
    "",
)


def _admin_dependencies_available() -> bool:
    return all(
        dependency is not None
        for dependency in (
            InMemoryTenantAdminRepository,
            TenantAdminSettingsService,
            TenantSecretCodec,
        )
    )


def _register_gemini_models(registry: ProviderRegistry, secrets_adapter: Any) -> None:
    api_key = secrets_adapter.get_optional_secret("GOOGLE_API_KEY")
    project = secrets_adapter.get_optional_secret("GOOGLE_CLOUD_PROJECT")
    location = secrets_adapter.get_optional_secret("GOOGLE_CLOUD_LOCATION") or "global"
    has_google_credentials = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")) or (
        load_google_credentials() is not None
    )
    if not api_key and not has_google_credentials:
        return
    for _title, model in config.MODEL_CANDIDATES:
        try:
            registry.register(
                model,
                GeminiAIAdapter(
                    api_key=api_key,
                    model=model,
                    project=project,
                    location=location,
                ),
            )
        except CallsAnalyserError:
            continue


def _build_model_options(ai_registry: ProviderRegistry) -> List[Tuple[str, str]]:
    if not config.PROJECT_IMPORTS_AVAILABLE:
        return []
    options: list[tuple[str, str]] = []
    for title, model_key in config.MODEL_CANDIDATES:
        if model_key not in ai_registry:
            continue
        provider = ai_registry.get(model_key)
        provider_label = getattr(provider, "provider_name", model_key)
        options.append((f"{provider_label} • {title}", model_key))
    return options


def _build_tenant_settings_repository(
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    *,
    codec: Any | None = None,
) -> Any:
    if supabase_url and supabase_key and SupabaseTenantSettingsRepository is not None:
        try:
            return SupabaseTenantSettingsRepository(
                supabase_url,
                supabase_key,
                codec=codec,
            )
        except Exception:
            pass
    return None


def _build_tenant_service(
    secrets_adapter: EnvSecretsAdapter,
    *,
    tenant_settings_source: Any | None = None,
) -> TenantService:
    return TenantService(
        secrets_adapter,
        default_tenant=config.DEFAULT_TENANT_ID,
        default_base_url=config.DEFAULT_BASE_URL,
        tenant_settings_source=tenant_settings_source,
    )


def _build_call_log_service(tenant_service: TenantService, storage_adapter: Any) -> CallLogService:
    del tenant_service
    factory = default_telephony_provider_factory()
    return CallLogService(factory.create, storage_adapter)


def _build_email_report_service() -> Any:
    if EmailReportService is None:
        return None

    recipient = os.environ.get("EMAIL_TO", "").strip() or GMAIL_ADDRESS

    if os.environ.get("BREVO_API_KEY", "").strip() and BrevoHTTPSAdapter is not None:
        return EmailReportService(
            BrevoHTTPSAdapter.from_env(),
            sender=os.environ.get("EMAIL_FROM", "").strip() or GMAIL_ADDRESS,
            recipient=recipient,
        )

    if os.environ.get("GOOGLE_app", "").strip() and GmailSMTPAdapter is not None:
        return EmailReportService(
            GmailSMTPAdapter.from_env(),
            sender=GMAIL_ADDRESS,
            recipient=recipient,
        )

    return None


def _build_auth_service(
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> Any:
    if AuthService is None or InMemoryAuthRepository is None or hash_password is None:
        return None

    if supabase_url and supabase_key and SupabaseAuthRepository is not None:
        try:
            return AuthService(SupabaseAuthRepository(supabase_url, supabase_key))
        except Exception:
            pass

    password = os.environ.get("VOCHI_UI_PASSWORD") or ""
    login = (os.environ.get("VOCHI_UI_LOGIN") or "").strip()

    users: list[dict[str, Any]] = []
    tenants: list[dict[str, Any]] = []
    access: list[dict[str, Any]] = []
    if password:
        login = login or "admin"
        user_id = f"local-ui-{login}"
        tenant_id = config.DEFAULT_TENANT_ID
        users.append(
            {
                "id": user_id,
                "login": login,
                "password_hash": hash_password(password),
                "display_name": login,
                "is_active": True,
            }
        )
        tenants.append(
            {
                "id": tenant_id,
                "display_name": tenant_id,
                "status": "active",
            }
        )
        access.append(
            {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": "admin",
            }
        )

    return AuthService(InMemoryAuthRepository(users=users, tenants=tenants, access=access))


def _build_tenant_settings_service(
    batch_params: BatchParams,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    *,
    repository: Any | None = None,
) -> Any:
    if TenantSettingsService is None or InMemoryTenantSettingsRepository is None:
        return None

    if repository is None:
        repository = _build_tenant_settings_repository(supabase_url, supabase_key)
    if repository is not None:
        return TenantSettingsService(
            repository,
            batch_params=batch_params,
            defaults=config,
        )

    return TenantSettingsService(
        InMemoryTenantSettingsRepository(settings={}, secrets={}),
        batch_params=batch_params,
        defaults=config,
    )


def _build_prompt_service(
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    *,
    repository: Any | None = None,
) -> Any:
    if PromptService is None:
        return None

    if repository is None and supabase_url and supabase_key and SupabasePromptTemplateRepository is not None:
        try:
            repository = SupabasePromptTemplateRepository(supabase_url, supabase_key)
        except Exception:
            repository = None

    if repository is None:
        return PromptService(config.PROMPTS)

    return PromptService(config.PROMPTS, prompt_repository=repository)


def build_dependencies() -> AppDependencies:
    """Prepare wiring for services used by the UI."""
    if not config.PROJECT_IMPORTS_AVAILABLE or not _admin_dependencies_available():
        batch_params = load_batch_params()
        auth_service = _build_auth_service()
        tenant_settings_service = _build_tenant_settings_service(batch_params)
        # minimal fallbacks that keep the UI responsive even without deps
        class MockAdapter:
            def get_optional_secret(self, _):  # pragma: no cover - simple stub
                return os.environ.get(_)

        model_options: List[Tuple[str, str]] = []
        model_choices = model_options or [MODEL_PLACEHOLDER_CHOICE]
        model_default = model_options[0][1] if model_options else MODEL_PLACEHOLDER_CHOICE[1]

        try:
            batch_language = config.Language(config.BATCH_LANGUAGE_CODE)
        except Exception:
            batch_language = config.Language.AUTO

        return AppDependencies(
            project_imports_available=False,
            secrets_adapter=MockAdapter(),
            storage_adapter=None,
            prompt_service=None,
            ai_registry={},
            tenant_service=None,
            call_log_service=None,
            analysis_service=None,
            usage_tracker=None,
            usage_report_repository=None,
            email_report_service=None,
            model_options=model_options,
            model_choices=model_choices,
            model_default=model_default,
            model_info="Add GOOGLE_API_KEY, GOOGLE_SERVICE_ACCOUNT_JSON_B64, or GOOGLE_APPLICATION_CREDENTIALS to enable models",
            batch_prompt_key=config.BATCH_PROMPT_KEY,
            batch_prompt_text=config.BATCH_PROMPT_TEXT,
            batch_model_key=config.BATCH_MODEL_KEY or model_default or "",
            batch_language=batch_language,
            batch_custom_conditions=config.BATCH_CUSTOM_CONDITIONS_DEFAULT,
            batch_custom_prompt_template=config.BATCH_CUSTOM_PROMPT_TEMPLATE,
            batch_params=batch_params,
            auth_service=auth_service,
            tenant_settings_service=tenant_settings_service,
        )

    secrets_adapter = EnvSecretsAdapter()
    storage_adapter = LocalStorageAdapter()
    ai_registry: ProviderRegistry[AIModelPort] = ProviderRegistry()
    batch_params = load_batch_params()
    supabase_url = secrets_adapter.get_optional_secret("SUPABASE_URL")
    supabase_key = secrets_adapter.get_optional_secret("SUPABASE_KEY")
    supabase_requested = bool(supabase_url or supabase_key)
    codec = TenantSecretCodec(secrets_adapter.get_optional_secret("TENANT_SECRETS_MASTER_KEY"))
    tenant_settings_repository = _build_tenant_settings_repository(
        supabase_url,
        supabase_key,
        codec=codec,
    )
    if tenant_settings_repository is None and not supabase_requested:
        tenant_id = config.DEFAULT_TENANT_ID
        tenant_settings_repository = InMemoryTenantAdminRepository(
            tenants={tenant_id: {"display_name": tenant_id, "status": "active"}},
            settings={},
            secrets={},
            prompts={},
            codec=codec,
        )
    if tenant_settings_repository is None:
        runtime_settings_repository = InMemoryTenantSettingsRepository(
            settings={},
            secrets={},
        )
        tenant_settings_source = None
        tenant_admin_settings_service = None
        prompt_service = _build_prompt_service(supabase_url, supabase_key)
    else:
        runtime_settings_repository = tenant_settings_repository
        tenant_settings_source = tenant_settings_repository
        tenant_admin_settings_service = TenantAdminSettingsService(
            tenant_settings_repository
        )
        prompt_service = _build_prompt_service(
            supabase_url,
            supabase_key,
            repository=tenant_settings_repository,
        )
    auth_service = _build_auth_service(supabase_url, supabase_key)
    tenant_settings_service = _build_tenant_settings_service(
        batch_params,
        supabase_url,
        supabase_key,
        repository=runtime_settings_repository,
    )

    _register_gemini_models(ai_registry, secrets_adapter)

    tenant_service = _build_tenant_service(
        secrets_adapter,
        tenant_settings_source=tenant_settings_source,
    )
    call_log_service = _build_call_log_service(tenant_service, storage_adapter)

    if supabase_url and supabase_key:
        cache = SupabaseCache(supabase_url, supabase_key)
        usage_tracker = SupabaseUsageTracker(supabase_url, supabase_key)
        usage_report_repository = SupabaseUsageReportRepository(supabase_url, supabase_key)
    else:
        cache_path = os.path.join(os.getcwd(), ".cache", "analysis_cache.json")
        cache = FileBackedCache(cache_path)
        usage_tracker = None
        usage_report_repository = None

    analysis_service = AnalysisService(
        call_log_service,
        ai_registry,
        prompt_service,
        cache=cache,
        usage_tracker=usage_tracker,
    )
    email_report_service = _build_email_report_service()

    model_options = _build_model_options(ai_registry)
    model_choices = model_options or [MODEL_PLACEHOLDER_CHOICE]
    model_default = model_options[0][1] if model_options else MODEL_PLACEHOLDER_CHOICE[1]
    model_info = (
        "Select an AI model for call analysis"
        if model_options
        else "Add GOOGLE_API_KEY, GOOGLE_SERVICE_ACCOUNT_JSON_B64, or GOOGLE_APPLICATION_CREDENTIALS to enable models"
    )

    try:
        batch_language = config.Language(config.BATCH_LANGUAGE_CODE)
    except ValueError:
        batch_language = config.Language.AUTO

    return AppDependencies(
        project_imports_available=True,
        secrets_adapter=secrets_adapter,
        storage_adapter=storage_adapter,
        prompt_service=prompt_service,
        ai_registry=ai_registry,
        tenant_service=tenant_service,
        call_log_service=call_log_service,
        analysis_service=analysis_service,
        usage_tracker=usage_tracker,
        usage_report_repository=usage_report_repository,
        email_report_service=email_report_service,
        model_options=model_options,
        model_choices=model_choices,
        model_default=model_default,
        model_info=model_info,
        batch_prompt_key=config.BATCH_PROMPT_KEY,
        batch_prompt_text=config.BATCH_PROMPT_TEXT,
        batch_model_key=config.BATCH_MODEL_KEY or model_default or "",
        batch_language=batch_language,
        batch_custom_conditions=config.BATCH_CUSTOM_CONDITIONS_DEFAULT,
        batch_custom_prompt_template=config.BATCH_CUSTOM_PROMPT_TEMPLATE,
        batch_params=batch_params,
        auth_service=auth_service,
        tenant_settings_service=tenant_settings_service,
        tenant_admin_settings_service=tenant_admin_settings_service,
    )
