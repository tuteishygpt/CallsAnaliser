"""FastAPI application exposing call and analysis endpoints."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from calls_analyser.domain.exceptions import CallsAnalyserError
from calls_analyser.domain.models import AnalysisResult, Language
from calls_analyser.services.analysis import AnalysisOptions, AnalysisService
from calls_analyser.services.call_log import CallLogService
from calls_analyser.services.prompt import PromptService
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.services.tenant import TenantConfig, TenantService
from calls_analyser.ports.ai import AIModelPort


class AnalysisRequest(BaseModel):
    """Request body for analysis endpoint."""

    prompt_key: str
    model_key: str
    custom_prompt: Optional[str] = None
    language: Language = Language.AUTO


class AnalysisResponse(BaseModel):
    """Response for analysis endpoint."""

    result: AnalysisResult


class CallLogResponse(BaseModel):
    """Call logs response."""

    data: list[dict[str, Any]]


def create_api_app(
    tenant_service: TenantService,
    call_log_service: CallLogService,
    analysis_service: AnalysisService,
    prompt_service: PromptService,
    ai_registry: ProviderRegistry[AIModelPort],
) -> FastAPI:
    """Create a configured FastAPI application."""

    app = FastAPI(title="Calls Analyser API")

    def resolve_tenant(tenant_id: Optional[str]) -> TenantConfig:
        try:
            return tenant_service.resolve(tenant_id)
        except CallsAnalyserError as exc:  # pragma: no cover - simple mapping
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/calls/{day}", response_model=CallLogResponse)
    def list_calls(day: date, tenant_id: Optional[str] = None) -> CallLogResponse:
        tenant = resolve_tenant(tenant_id)
        calls = call_log_service.list_calls(day, tenant)
        return CallLogResponse(data=[entry.raw for entry in calls])

    @app.post("/analysis/{unique_id}", response_model=AnalysisResponse)
    def analyze(unique_id: str, req: AnalysisRequest, tenant_id: Optional[str] = None) -> AnalysisResponse:
        tenant = resolve_tenant(tenant_id)
        result = analysis_service.analyze_call(
            unique_id=unique_id,
            tenant=tenant,
            lang=req.language,
            options=AnalysisOptions(
                model_key=req.model_key,
                prompt_key=req.prompt_key,
                custom_prompt=req.custom_prompt,
            ),
        )
        return AnalysisResponse(result=result)

    @app.get("/prompts")
    def prompts() -> Dict[str, str]:
        return {tpl.key: tpl.title for tpl in prompt_service.list_templates().values()}

    @app.get("/models")
    def models() -> Dict[str, str]:
        return {key: provider.provider_name for key, provider in ai_registry.items()}

    return app
