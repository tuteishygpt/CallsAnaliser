"""FastAPI application exposing call and analysis endpoints."""
from __future__ import annotations

import base64
import binascii
from datetime import date
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
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
    auth_service: Optional[Any] = None,
) -> FastAPI:
    """Create a configured FastAPI application."""

    app = FastAPI(title="Calls Analyser API")

    def authenticated_user_id(request: Request) -> str:
        authorization = request.headers.get("Authorization")
        if authorization:
            scheme, _, credentials = authorization.partition(" ")
            credentials = credentials.strip()
            if scheme.lower() == "basic" and credentials:
                return authenticated_basic_user_id(credentials)
            if scheme.lower() == "bearer" and credentials:
                return authenticated_bearer_user_id(credentials)

        raise HTTPException(status_code=401, detail="Authentication required")

    def authenticated_basic_user_id(credentials: str) -> str:
        authenticate = getattr(auth_service, "authenticate", None)
        if not callable(authenticate):
            raise HTTPException(status_code=401, detail="Authentication required")

        try:
            decoded = base64.b64decode(credentials.encode("ascii"), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise HTTPException(status_code=401, detail="Authentication required")

        login, separator, password = decoded.partition(":")
        if not separator:
            raise HTTPException(status_code=401, detail="Authentication required")

        user = authenticate(login, password)
        user_id = getattr(user, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        return str(user_id)

    def authenticated_bearer_user_id(token: str) -> str:
        authenticate_token = getattr(auth_service, "authenticate_token", None)
        if not callable(authenticate_token):
            raise HTTPException(status_code=401, detail="Authentication required")

        user = authenticate_token(token)
        user_id = getattr(user, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        return str(user_id)

    def resolve_tenant(tenant_id: Optional[str], request: Request) -> TenantConfig:
        user_id: Optional[str] = None
        if auth_service is not None:
            user_id = authenticated_user_id(request)

        try:
            tenant = tenant_service.resolve(tenant_id)
        except CallsAnalyserError as exc:  # pragma: no cover - simple mapping
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if auth_service is not None and user_id is not None:
            if not auth_service.can_access_tenant(user_id, tenant.tenant_id):
                raise HTTPException(status_code=403, detail="Tenant access denied")

        return tenant

    @app.get("/calls/{day}", response_model=CallLogResponse)
    def list_calls(day: date, request: Request, tenant_id: Optional[str] = None) -> CallLogResponse:
        tenant = resolve_tenant(tenant_id, request)
        calls = call_log_service.list_calls(day, tenant)
        return CallLogResponse(data=[entry.raw for entry in calls])

    @app.post("/analysis/{unique_id}", response_model=AnalysisResponse)
    def analyze(
        unique_id: str,
        req: AnalysisRequest,
        request: Request,
        tenant_id: Optional[str] = None,
    ) -> AnalysisResponse:
        tenant = resolve_tenant(tenant_id, request)
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
