"""Dependency wiring for the Gradio UI."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple

from . import config
from calls_analyser.batch_params import BatchParams, load_batch_params

try:  # pragma: no cover - optional imports
    from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
    from calls_analyser.adapters.secrets.env import EnvSecretsAdapter
    from calls_analyser.adapters.storage.local import LocalStorageAdapter
    from calls_analyser.adapters.storage.supabase_storage import SupabaseCache
    from calls_analyser.adapters.telephony.mts_vats import MtsVatsTelephonyAdapter
    from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter
    from calls_analyser.domain.exceptions import CallsAnalyserError
    from calls_analyser.ports.ai import AIModelPort
    from calls_analyser.services.analysis import AnalysisOptions, AnalysisService
    from calls_analyser.services.cache import FileBackedCache
    from calls_analyser.services.call_log import CallLogService
    from calls_analyser.services.prompt import PromptService
    from calls_analyser.services.registry import ProviderRegistry
    from calls_analyser.services.tenant import TenantService
except ImportError:  # pragma: no cover - executed when project deps unavailable
    GeminiAIAdapter = None  # type: ignore
    EnvSecretsAdapter = None  # type: ignore
    LocalStorageAdapter = None  # type: ignore
    SupabaseCache = None  # type: ignore
    MtsVatsTelephonyAdapter = None  # type: ignore
    VochiTelephonyAdapter = None  # type: ignore
    CallsAnalyserError = Exception  # type: ignore
    AIModelPort = Any  # type: ignore
    AnalysisOptions = None  # type: ignore
    AnalysisOptions = None  # type: ignore
    AnalysisService = None  # type: ignore
    FileBackedCache = None  # type: ignore
    CallLogService = None  # type: ignore
    PromptService = None  # type: ignore
    ProviderRegistry = Dict  # type: ignore
    TenantService = None  # type: ignore


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


MODEL_PLACEHOLDER_CHOICE = (
    "Configure GOOGLE_API_KEY to enable Gemini models",
    "",
)


def _register_gemini_models(registry: ProviderRegistry, secrets_adapter: Any) -> None:
    api_key = secrets_adapter.get_optional_secret("GOOGLE_API_KEY")
    if not api_key:
        return
    for _title, model in config.MODEL_CANDIDATES:
        try:
            registry.register(
                model,
                GeminiAIAdapter(
                    api_key=api_key,
                    model=model,
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


def _build_tenant_service(secrets_adapter: EnvSecretsAdapter) -> TenantService:
    return TenantService(
        secrets_adapter,
        default_tenant=config.DEFAULT_TENANT_ID,
        default_base_url=config.DEFAULT_BASE_URL,
    )


def _build_call_log_service(tenant_service: TenantService, storage_adapter: Any) -> CallLogService:
    config_obj = tenant_service.resolve()
    if config_obj.provider == "mts_vats":
        telephony_adapter = MtsVatsTelephonyAdapter(
            domain=config_obj.mts_domain or config_obj.vochi_base_url,
            api_key=config_obj.mts_api_key or "",
        )
    else:
        telephony_adapter = VochiTelephonyAdapter(
            base_url=config_obj.vochi_base_url,
            client_id=config_obj.vochi_client_id,
            bearer_token=config_obj.bearer_token,
        )
    return CallLogService(telephony_adapter, storage_adapter)


def build_dependencies() -> AppDependencies:
    """Prepare wiring for services used by the UI."""
    if not config.PROJECT_IMPORTS_AVAILABLE:
        batch_params = load_batch_params()
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
            model_options=model_options,
            model_choices=model_choices,
            model_default=model_default,
            model_info="Add GOOGLE_API_KEY to secrets and reload to enable models",
            batch_prompt_key=config.BATCH_PROMPT_KEY,
            batch_prompt_text=config.BATCH_PROMPT_TEXT,
            batch_model_key=config.BATCH_MODEL_KEY or model_default or "",
            batch_language=batch_language,
            batch_custom_conditions=config.BATCH_CUSTOM_CONDITIONS_DEFAULT,
            batch_custom_prompt_template=config.BATCH_CUSTOM_PROMPT_TEMPLATE,
            batch_params=batch_params,
        )

    secrets_adapter = EnvSecretsAdapter()
    storage_adapter = LocalStorageAdapter()
    prompt_service = PromptService(config.PROMPTS)
    ai_registry: ProviderRegistry[AIModelPort] = ProviderRegistry()

    _register_gemini_models(ai_registry, secrets_adapter)

    tenant_service = _build_tenant_service(secrets_adapter)
    call_log_service = _build_call_log_service(tenant_service, storage_adapter)

    supabase_url = secrets_adapter.get_optional_secret("SUPABASE_URL")
    supabase_key = secrets_adapter.get_optional_secret("SUPABASE_KEY")

    if supabase_url and supabase_key:
        cache = SupabaseCache(supabase_url, supabase_key)
    else:
        cache_path = os.path.join(os.getcwd(), ".cache", "analysis_cache.json")
        cache = FileBackedCache(cache_path)

    analysis_service = AnalysisService(call_log_service, ai_registry, prompt_service, cache=cache)

    model_options = _build_model_options(ai_registry)
    model_choices = model_options or [MODEL_PLACEHOLDER_CHOICE]
    model_default = model_options[0][1] if model_options else MODEL_PLACEHOLDER_CHOICE[1]
    model_info = (
        "Select an AI model for call analysis"
        if model_options
        else "Add GOOGLE_API_KEY to secrets and reload to enable models"
    )

    try:
        batch_language = config.Language(config.BATCH_LANGUAGE_CODE)
    except ValueError:
        batch_language = config.Language.AUTO

    batch_params = load_batch_params()

    return AppDependencies(
        project_imports_available=True,
        secrets_adapter=secrets_adapter,
        storage_adapter=storage_adapter,
        prompt_service=prompt_service,
        ai_registry=ai_registry,
        tenant_service=tenant_service,
        call_log_service=call_log_service,
        analysis_service=analysis_service,
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
    )
