"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import datetime as _dt
import os
from typing import Optional

import gradio as gr
import pandas as pd

from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
from calls_analyser.adapters.secrets.env import EnvSecretsAdapter
from calls_analyser.adapters.storage.local import LocalStorageAdapter
from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter
from calls_analyser.domain.exceptions import CallsAnalyserError
from calls_analyser.domain.models import Language
from calls_analyser.ports.ai import AIModelPort
from calls_analyser.services.analysis import AnalysisOptions, AnalysisService
from calls_analyser.services.call_log import CallLogService
from calls_analyser.services.prompt import PromptService, PromptTemplate
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.services.tenant import TenantService


PROMPTS = {
    "simple": PromptTemplate(
        key="simple",
        title="Simple",
        body=(
            "You are a call-center conversation analyst for a medical clinic. From the call recording, provide a brief summary:\n"
            "- Purpose of the call (appointment / results / complaint / billing / other).\n"
            "- Patient intent and expectations.\n"
            "- Outcome (booked / call-back / routed / unresolved).\n"
            "- Next steps (owner and when).\n"
            "- Patient emotion (1–5) and agent tone (1–5).\n"
            "- Alerts: urgency/risks/privacy.\n\n"
            "Keep it short (6–8 lines). End with a line: ‘Service quality rating: X/5’ and one sentence explaining the rating."
        ),
    ),
    "medium": PromptTemplate(
        key="medium",
        title="Medium",
        body=(
            "Act as a senior service analyst. Analyze the call using this structure:\n"
            "1) Quick overview: reason for the call, intent, key facts, urgency (low/medium/high).\n"
            "2) Call flow (2–4 bullets): what was asked/answered, where friction occurred.\n"
            "3) Outcomes & tasks: concrete next actions for clinic/patient with timeframes.\n"
            "4) Emotions & empathy: patient mood; agent empathy (0–5).\n"
            "5) Procedural compliance: identity verification, disclosure of recording (if stated), no off-protocol medical advice, data accuracy.\n"
            "6) Quality rating (0–100) using rubric: greeting, verification, accuracy, empathy, issue resolution (each 0–20)."
        ),
    ),
    "detailed": PromptTemplate(
        key="detailed",
        title="Detailed",
        body=(
            "You are a quality & operations analyst. Provide an in-depth analysis:\n"
            "A) Segmentation: split the call into stages with approximate timestamps (if available) and roles (Patient/Agent).\n"
            "B) Structured data for booking: full name (if stated), date of birth, phone, symptoms/complaints (list), onset/duration, possible pain level 0–10 (if mentioned), required specialist/service, preferred time windows, constraints.\n"
            "C) Triage & risks: class (routine/urgent/emergency), red flags, whether immediate escalation is needed.\n"
            "D) Compliance audit: identity/privacy checks, recording disclosure, consent to data processing, booking policies.\n"
            "E) Conversation metrics: talk ratio (agent/patient), interruptions, long pauses, notable keywords.\n"
            "F) Coaching for the agent: 3–5 concrete improvements with sample phrasing.\n\n"
            "Deliver: (1) A short patient-chart summary (2–3 sentences). (2) A task table with columns: priority, owner, due."
        ),
    ),
}

TPL_OPTIONS = [(tpl.title, tpl.key) for tpl in PROMPTS.values()] + [("Custom", "custom")]
LANG_OPTIONS = [
    ("Russian", Language.RUSSIAN.value),
    ("Auto", Language.AUTO.value),
    ("Belarusian", Language.BELARUSIAN.value),
    ("English", Language.ENGLISH.value),
]
MODEL_OPTIONS = [
    ("flash", "models/gemini-2.5-flash"),
    ("pro", "models/gemini-2.5-pro"),
    ("flash-lite", "models/gemini-2.5-flash-lite"),
]


# ----------------------------------------------------------------------------
# Dependency wiring
# ----------------------------------------------------------------------------
DEFAULT_TENANT_ID = os.environ.get("DEFAULT_TENANT_ID", "default")
DEFAULT_BASE_URL = os.environ.get("VOCHI_BASE_URL", "https://crm.vochi.by/api")
secrets_adapter = EnvSecretsAdapter()
storage_adapter = LocalStorageAdapter()
prompt_service = PromptService(PROMPTS)
ai_registry: ProviderRegistry[AIModelPort] = ProviderRegistry()

def _register_gemini_models() -> None:
    api_key = secrets_adapter.get_optional_secret("GOOGLE_API_KEY")
    if not api_key:
        return
    for _title, model in MODEL_OPTIONS:
        try:
            ai_registry.register(model, GeminiAIAdapter(api_key=api_key, model=model))
        except CallsAnalyserError:
            # Skip registration if dependency is missing
            continue


_register_gemini_models()

def _build_tenant_service() -> TenantService:
    return TenantService(
        secrets_adapter,
        default_tenant=DEFAULT_TENANT_ID,
        default_base_url=DEFAULT_BASE_URL,
    )


def _build_call_log_service(tenant_service: TenantService) -> CallLogService:
    config = tenant_service.resolve()
    telephony_adapter = VochiTelephonyAdapter(
        base_url=config.vochi_base_url,
        client_id=config.vochi_client_id,
        bearer_token=config.bearer_token,
    )
    return CallLogService(telephony_adapter, storage_adapter)


tenant_service = _build_tenant_service()
call_log_service = _build_call_log_service(tenant_service)
analysis_service = AnalysisService(call_log_service, ai_registry, prompt_service)


# ----------------------------------------------------------------------------
# UI utilities
# ----------------------------------------------------------------------------

def _label_row(row: dict) -> str:
    start = row.get("Start", "")
    src = row.get("CallerId", "")
    dst = row.get("Destination", "")
    dur = row.get("Duration", "")
    return f"{start} | {src} → {dst} ({dur}s)"


def _parse_day(day_str: str) -> _dt.date:
    return _dt.date.fromisoformat(day_str.strip())


# ----------------------------------------------------------------------------
# Gradio handlers
# ----------------------------------------------------------------------------

def ui_fetch_calls(date_str: str, tenant_id: str):
    try:
        day = _parse_day(date_str)
        tenant = tenant_service.resolve(tenant_id or None)
        entries = call_log_service.list_calls(day, tenant)
        data = [entry.raw for entry in entries]
        df = pd.DataFrame(data)
        opts = [( _label_row(row), idx) for idx, row in df.iterrows()]
        dd = gr.update(choices=[(label, idx) for label, idx in opts], value=(opts[0][1] if opts else None))
        msg = f"Calls found: {len(df)}"
        return df, dd, msg
    except CallsAnalyserError as exc:
        return pd.DataFrame(), gr.update(choices=[], value=None), f"Domain error: {exc}"
    except Exception as exc:
        return pd.DataFrame(), gr.update(choices=[], value=None), f"Load error: {exc}"


def ui_play_audio(selected_idx: Optional[int], df: pd.DataFrame, tenant_id: str):
    if selected_idx is None or df is None or df.empty:
        return "<em>First fetch the list and select a row.</em>", None, None, ""
    try:
        row = df.iloc[int(selected_idx)]
    except Exception:
        return "<em>Invalid row selection.</em>", None, None, ""
    unique_id = str(row.get("UniqueId"))
    if not unique_id:
        return "<em>Selected row has no UniqueId.</em>", None, None, ""
    try:
        tenant = tenant_service.resolve(tenant_id or None)
        handle = call_log_service.ensure_recording(unique_id, tenant)
        html = f'URL: <a href="{handle.source_uri}" target="_blank">{handle.source_uri}</a>'
        return html, handle.local_uri, handle.local_uri, "Ready ✅"
    except CallsAnalyserError as exc:
        return f"Playback failed: {exc}", None, None, ""
    except Exception as exc:
        return f"Playback failed: {exc}", None, None, ""


def ui_toggle_custom_prompt(template_key: str):
    return gr.update(visible=(template_key == "custom"))


def ui_analyze(
    selected_idx: Optional[int],
    df: pd.DataFrame,
    template_key: str,
    custom_prompt: str,
    lang_code: str,
    model_pref: str,
    tenant_id: str,
):
    if df is None or df.empty or selected_idx is None:
        return "First fetch the list, choose a call, and (optionally) click ‘🎧 Play’."
    if model_pref not in ai_registry:
        return "❌ Selected model is not available. Check API key or provider configuration."
    try:
        row = df.iloc[int(selected_idx)]
    except Exception:
        return "Invalid row selection."
    unique_id = str(row.get("UniqueId"))
    if not unique_id:
        return "Selected row has no UniqueId."
    try:
        tenant = tenant_service.resolve(tenant_id or None)
        lang = Language(lang_code)
        result = analysis_service.analyze_call(
            unique_id=unique_id,
            tenant=tenant,
            lang=lang,
            options=AnalysisOptions(
                model_key=model_pref,
                prompt_key=template_key,
                custom_prompt=custom_prompt,
            ),
        )
        return f"### Analysis result\n\n{result.text}"
    except CallsAnalyserError as exc:
        return f"Analysis failed: {exc}"
    except Exception as exc:
        return f"Analysis failed: {exc}"


def ui_check_password(pwd: str):
    if not _UI_PASSWORD:
        return False, (
            "⚠️ <b>VOCHI_UI_PASSWORD</b> не наладжаны ў Secrets. "
            "Дадайце яго ў Settings → Secrets і перазапусціце Space."
        ), gr.update(visible=True)
    if (pwd or "").strip() == _UI_PASSWORD:
        return True, "✅ Доступ адкрыты. Цяпер можна націскаць <b>Fetch list</b> і працаваць.", gr.update(visible=False)
    return False, "❌ Няправільны пароль. Паспрабуйце яшчэ раз.", gr.update(visible=True)


def ui_fetch_or_auth(date_str: str, authed: bool, tenant_id: str):
    if not authed:
        return gr.update(), gr.update(), "🔒 Увядзіце пароль, каб атрымаць званкі.", gr.update(visible=True)
    df, dd, msg = ui_fetch_calls(date_str, tenant_id)
    return df, dd, msg, gr.update(visible=False)


# ----------------------------------------------------------------------------
# Password config
# ----------------------------------------------------------------------------
_UI_PASSWORD = os.environ.get("VOCHI_UI_PASSWORD", "")


# ----------------------------------------------------------------------------
# Build Gradio UI
# ----------------------------------------------------------------------------

def _today_str():
    return _dt.date.today().strftime("%Y-%m-%d")


with gr.Blocks(title="Vochi CRM Call Logs (Gradio)") as demo:
    gr.Markdown(
        """
        # Vochi CRM → MP3 → AI analysis
        *Fetch daily calls, play/download MP3, and analyze the call with an AI model.*

        """
    )

    authed = gr.State(False)

    with gr.Group(visible=False) as pwd_group:
        gr.Markdown("### 🔐 Увядзіце пароль")
        pwd_tb = gr.Textbox(label="Password", type="password", placeholder="••••••••", lines=1)
        pwd_btn = gr.Button("Адкрыць доступ", variant="primary")

    with gr.Tabs() as tabs:
        with gr.Tab("Vochi CRM"):
            with gr.Row():
                tenant_tb = gr.Textbox(label="Tenant ID", value=DEFAULT_TENANT_ID, scale=1)
                date_inp = gr.Textbox(label="Date", value=_today_str(), scale=1)
                fetch_btn = gr.Button("Fetch list", variant="primary", scale=0)
            calls_df = gr.Dataframe(value=pd.DataFrame(), label="Call list", interactive=False)
            row_dd = gr.Dropdown(choices=[], label="Call", info="Select a row for playback/analysis")
            with gr.Row():
                play_btn = gr.Button("🎧 Play")
            url_html = gr.HTML()
            audio_out = gr.Audio(label="Audio", type="filepath")
            file_out = gr.File(label="MP3 download")
            status_fetch = gr.Markdown()

        with gr.Tab("AI Analysis"):
            with gr.Row():
                tpl_dd = gr.Dropdown(choices=TPL_OPTIONS, value="simple", label="Template")
                lang_dd = gr.Dropdown(choices=LANG_OPTIONS, value=Language.AUTO.value, label="Language")
                model_dd = gr.Dropdown(choices=MODEL_OPTIONS, value="models/gemini-2.5-flash", label="Model")
            custom_prompt_tb = gr.Textbox(label="Custom prompt", lines=8, visible=False)
            analyze_btn = gr.Button("🧠 Analyze", variant="primary")
            analysis_md = gr.Markdown()

    fetch_btn.click(
        ui_fetch_or_auth,
        inputs=[date_inp, authed, tenant_tb],
        outputs=[calls_df, row_dd, status_fetch, pwd_group],
    )

    pwd_btn.click(
        ui_check_password,
        inputs=[pwd_tb],
        outputs=[authed, status_fetch, pwd_group],
    )

    play_btn.click(
        ui_play_audio,
        inputs=[row_dd, calls_df, tenant_tb],
        outputs=[url_html, audio_out, file_out, status_fetch],
    )

    tpl_dd.change(ui_toggle_custom_prompt, inputs=[tpl_dd], outputs=[custom_prompt_tb])

    analyze_btn.click(
        ui_analyze,
        inputs=[row_dd, calls_df, tpl_dd, custom_prompt_tb, lang_dd, model_dd, tenant_tb],
        outputs=[analysis_md],
    )


if __name__ == "__main__":
    demo.launch()
