"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
from typing import Optional
import json

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
from calls_analyser.config import (
    PROMPTS as CFG_PROMPTS,
    MODEL_CANDIDATES as CFG_MODEL_CANDIDATES,
    BATCH_MODEL_KEY as CFG_BATCH_MODEL_KEY,
    BATCH_PROMPT_KEY as CFG_BATCH_PROMPT_KEY,
    BATCH_PROMPT_TEXT as CFG_BATCH_PROMPT_TEXT,
    BATCH_LANGUAGE_CODE as CFG_BATCH_LANGUAGE_CODE,
)


PROMPTS = CFG_PROMPTS

TPL_OPTIONS = [(tpl.title, tpl.key) for tpl in PROMPTS.values()] + [("Custom", "custom")]
LANG_OPTIONS = [
    ("Russian", Language.RUSSIAN.value),
    ("Auto", Language.AUTO.value),
    ("Belarusian", Language.BELARUSIAN.value),
    ("English", Language.ENGLISH.value),
]
CALL_TYPE_OPTIONS = [
    ("All types", ""),
    ("Inbound", "0"),
    ("Outbound", "1"),
    ("Internal", "2"),
]
MODEL_CANDIDATES = CFG_MODEL_CANDIDATES


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
    for _title, model in MODEL_CANDIDATES:
        try:
            ai_registry.register(model, GeminiAIAdapter(api_key=api_key, model=model))
        except CallsAnalyserError:
            # Skip registration if dependency is missing
            continue


_register_gemini_models()


def _build_model_options() -> list[tuple[str, str]]:
    """Return dropdown options for configured AI models."""

    options: list[tuple[str, str]] = []
    for title, model_key in MODEL_CANDIDATES:
        if model_key not in ai_registry:
            continue
        provider = ai_registry.get(model_key)
        provider_label = getattr(provider, "provider_name", model_key)
        options.append((f"{provider_label} • {title}", model_key))
    return options


MODEL_OPTIONS = _build_model_options()
MODEL_PLACEHOLDER_CHOICE = (
    "Configure GOOGLE_API_KEY to enable Gemini models",
    "",
)
MODEL_CHOICES = MODEL_OPTIONS or [MODEL_PLACEHOLDER_CHOICE]
MODEL_DEFAULT = MODEL_OPTIONS[0][1] if MODEL_OPTIONS else MODEL_PLACEHOLDER_CHOICE[1]
MODEL_INFO = (
    "Select an AI model for call analysis"
    if MODEL_OPTIONS
    else "Add GOOGLE_API_KEY to secrets and reload to enable models"
)

# Batch settings from config module
BATCH_PROMPT_KEY = CFG_BATCH_PROMPT_KEY
BATCH_PROMPT_TEXT = (CFG_BATCH_PROMPT_TEXT or "").strip()
BATCH_MODEL_KEY = CFG_BATCH_MODEL_KEY or MODEL_DEFAULT or ""
BATCH_LANGUAGE_CODE = CFG_BATCH_LANGUAGE_CODE
try:
    BATCH_LANGUAGE = Language(BATCH_LANGUAGE_CODE)
except ValueError:
    BATCH_LANGUAGE = Language.AUTO

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


def _parse_day(day_value) -> _dt.date:
    if isinstance(day_value, _dt.datetime):
        return day_value.date()
    if isinstance(day_value, _dt.date):
        return day_value
    if not day_value:
        raise ValueError("Date not specified.")
    
    # Handle Unix timestamp (float) from Gradio DateTime component
    try:
        timestamp = float(str(day_value).strip())
        # Unix timestamps are typically in seconds, check if it's reasonable
        if timestamp > 1e9:  # Unix timestamp for dates after 2001
            # Use UTC to avoid timezone issues
            return _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc).date()
    except (ValueError, TypeError):
        pass
    
    # Try to parse as ISO format string
    try:
        return _dt.date.fromisoformat(str(day_value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid date format: {day_value}") from exc


def _parse_time_value(time_value) -> Optional[_dt.time]:
    if time_value in (None, ""):
        return None
    if isinstance(time_value, _dt.datetime):
        return time_value.time().replace(microsecond=0)
    if isinstance(time_value, _dt.time):
        return time_value.replace(microsecond=0)
    
    # Handle potential Unix timestamp (though less likely for time inputs)
    try:
        timestamp = float(str(time_value).strip())
        if timestamp > 1e9:  # Unix timestamp
            # Use UTC to avoid timezone issues
            return _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc).time().replace(microsecond=0)
    except (ValueError, TypeError):
        pass
    
    value = str(time_value).strip()
    if not value:
        return None
    try:
        # Accept 'H:MM' by padding leading zero for hour
        if value.count(":") == 1 and len(value.split(":")[0]) == 1:
            value = f"0{value}"
        parsed = _dt.time.fromisoformat(value)
    except ValueError as exc:
        if len(value) == 5 and value.count(":") == 1:
            parsed = _dt.time.fromisoformat(f"{value}:00")
        else:
            raise ValueError(f"Invalid time format: {value}") from exc
    return parsed.replace(microsecond=0)


def _validate_time_range(time_from: Optional[_dt.time], time_to: Optional[_dt.time]) -> None:
    if time_from and time_to and time_from > time_to:
        raise ValueError("Time 'from' must be less than or equal to time 'to'.")


def _resolve_call_type(value: object) -> Optional[int]:
    s = str(value).strip()
    if s == "":
        return None
    # Try numeric directly
    try:
        return int(s)
    except ValueError:
        pass
    # Fallback: map label -> value using CALL_TYPE_OPTIONS
    label_to_value = {label: v for (label, v) in CALL_TYPE_OPTIONS}
    mapped = label_to_value.get(s, "")
    try:
        return int(mapped) if mapped != "" else None
    except ValueError:
        return None


def _build_dropdown(df: pd.DataFrame) -> gr.Update:
    opts = [( _label_row(row), idx) for idx, row in df.iterrows()]
    value = opts[0][1] if opts else None
    return gr.update(choices=[(label, idx) for label, idx in opts], value=value)


def _result_table_html(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "<em>No processed calls.</em>"
    df = pd.DataFrame(rows)
    return df.to_html(index=False, escape=False)


# ----------------------------------------------------------------------------
# Gradio handlers
# ----------------------------------------------------------------------------

def ui_filter_calls(
    date_value,
    time_from_value,
    time_to_value,
    call_type_value: str,
    authed: bool,
    tenant_id: str,
):
    if not authed:
        return (
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            "🔒 Enter the password to apply the filter.",
            gr.update(visible=True),
        )
    try:
        day = _parse_day(date_value)
        time_from = _parse_time_value(time_from_value)
        time_to = _parse_time_value(time_to_value)
        _validate_time_range(time_from, time_to)
        call_type = _resolve_call_type(call_type_value)
        tenant = tenant_service.resolve(tenant_id or None)
        entries = call_log_service.list_calls(
            day,
            tenant,
            time_from=time_from,
            time_to=time_to,
            call_type=call_type,
        )
        data = [entry.raw for entry in entries]
        df = pd.DataFrame(data)
        dd = _build_dropdown(df)
        msg = f"Calls found: {len(df)}"
        return df, dd, msg, gr.update(visible=False)
    except CallsAnalyserError as exc:
        return (
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            f"Domain error: {exc}",
            gr.update(visible=False),
        )
    except Exception as exc:
        return (
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            f"Load error: {exc}",
            gr.update(visible=False),
        )


def ui_play_audio(selected_idx: Optional[int], df: pd.DataFrame, tenant_id: str):
    if selected_idx is None or df is None or df.empty:
        return "<em>First fetch the list and select a row.</em>", None, ""
    try:
        row = df.iloc[int(selected_idx)]
    except Exception:
        return "<em>Invalid row selection.</em>", None, ""
    unique_id = str(row.get("UniqueId"))
    if not unique_id:
        return "<em>Selected row has no UniqueId.</em>", None, ""
    try:
        tenant = tenant_service.resolve(tenant_id or None)
        handle = call_log_service.ensure_recording(unique_id, tenant)
        listen_url = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{unique_id}"
        html = f'URL: <a href="{listen_url}" target="_blank">{listen_url}</a>'
        return html, handle.local_uri, "Ready ✅"
    except CallsAnalyserError as exc:
        return f"Playback failed: {exc}", None, ""
    except Exception as exc:
        return f"Playback failed: {exc}", None, ""


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
    if len(ai_registry) == 0:
        return "❌ No AI models are configured. Add provider credentials and reload the app."
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


def ui_mass_analyze(
    date_value,
    time_from_value,
    time_to_value,
    call_type_value: str,
    tenant_id: str,
    authed: bool,
    progress=gr.Progress(track_tqdm=False),
):
    empty_state = pd.DataFrame()
    reset_file = gr.update(value=None, visible=False)
    progress(0, desc="Preparing")
    if not authed:
        return empty_state, _result_table_html([]), "", "🔒 Enter the password to run batch analysis.", reset_file
    if len(ai_registry) == 0 or not BATCH_MODEL_KEY:
        return empty_state, _result_table_html([]), "", "❌ Batch analysis is unavailable: AI model is not configured.", reset_file
    if BATCH_MODEL_KEY not in ai_registry:
        return empty_state, _result_table_html([]), "", "❌ Selected model for batch analysis is unavailable.", reset_file
    try:
        day = _parse_day(date_value)
        time_from = _parse_time_value(time_from_value)
        time_to = _parse_time_value(time_to_value)
        _validate_time_range(time_from, time_to)
        call_type = _resolve_call_type(call_type_value)
        tenant = tenant_service.resolve(tenant_id or None)
        entries = call_log_service.list_calls(
            day,
            tenant,
            time_from=time_from,
            time_to=time_to,
            call_type=call_type,
        )
        total = len(entries)
        if total == 0:
            return (
                empty_state,
                _result_table_html([]),
                "Found: 0, processed: 0",
                "ℹ️ No calls for the selected filter.",
                reset_file,
            )

        rows: list[dict[str, object]] = []
        success = 0
        for idx, entry in enumerate(entries):
            progress(idx / total, desc=f"Analyzing {idx + 1}/{total}")
            row_data: dict[str, object] = {
                "Start": entry.started_at.isoformat() if entry.started_at else entry.raw.get("Start", ""),
                "Caller": entry.caller_id or "",
                "Destination": entry.destination or "",
                "Duration (s)": entry.duration_seconds,
                "UniqueId": entry.unique_id,
            }
            handle = None
            try:
                handle = call_log_service.ensure_recording(entry.unique_id, tenant)
                result = analysis_service.analyze_call(
                    unique_id=entry.unique_id,
                    tenant=tenant,
                    lang=BATCH_LANGUAGE,
                    options=AnalysisOptions(
                        model_key=BATCH_MODEL_KEY,
                        prompt_key=BATCH_PROMPT_KEY,
                        custom_prompt=BATCH_PROMPT_TEXT or None,
                    ),
                )
                link = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{entry.unique_id}"
                # Parse model JSON response into two columns
                try:
                    text = str(result.text or "").strip()
                    # Extract JSON object if wrapped (e.g., code fences or prefix/suffix text)
                    l = text.find("{")
                    r = text.rfind("}")
                    if l != -1 and r != -1 and r > l:
                        text = text[l : r + 1]
                    payload = json.loads(text)
                    needs = bool(payload.get("needs_follow_up"))
                    reason = str(payload.get("reason") or "")
                    row_data["Needs follow-up"] = "Yes" if needs else "No"
                    row_data["Reason"] = reason
                except Exception:
                    # Fallback: keep raw text in reason if JSON parse failed
                    row_data["Needs follow-up"] = ""
                    row_data["Reason"] = result.text
                row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>' if link else ""
                row_data["Status"] = "✅"
                success += 1
            except CallsAnalyserError as exc:
                link = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{entry.unique_id}"
                row_data["Needs follow-up"] = ""
                row_data["Reason"] = f"❌ {exc}"
                row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>' if link else ""
                row_data["Status"] = "❌"
            except Exception as exc:
                link = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{entry.unique_id}"
                row_data["Needs follow-up"] = ""
                row_data["Reason"] = f"❌ {exc}"
                row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>' if link else ""
                row_data["Status"] = "❌"
            rows.append(row_data)
            progress((idx + 1) / total, desc=f"Analyzing {idx + 1}/{total}")

        df = pd.DataFrame(rows)
        summary = f"Found: {total}, processed: {success}"
        status = "✅ Batch analysis completed."
        return df, _result_table_html(rows), summary, status, reset_file
    except CallsAnalyserError as exc:
        return empty_state, _result_table_html([]), "", f"Analysis failed: {exc}", reset_file
    except Exception as exc:
        return empty_state, _result_table_html([]), "", f"Analysis failed: {exc}", reset_file


def ui_export_results(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        return gr.update(value=None, visible=False), "❌ No data to export."
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        results_df.to_csv(tmp.name, index=False)
        file_path = tmp.name
    return gr.update(value=file_path, visible=True), "✅ File is ready to save."


def ui_check_password(pwd: str):
    if not _UI_PASSWORD:
        return False, (
            "⚠️ <b>VOCHI_UI_PASSWORD</b> is not configured in Secrets. "
            "Add it in Settings → Secrets and reload the Space."
        ), gr.update(visible=True)
    if (pwd or "").strip() == _UI_PASSWORD:
        return True, "✅ Access granted. You can now click <b>Filter</b> and proceed.", gr.update(visible=False)
    return False, "❌ Incorrect password. Please try again.", gr.update(visible=True)


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
        *Filter calls by date, time and type, listen to recordings and run batch AI analysis.*

        """
    )

    authed = gr.State(False)
    batch_results_state = gr.State(pd.DataFrame())

    with gr.Group(visible=False) as pwd_group:
        gr.Markdown("### 🔐 Enter password")
        pwd_tb = gr.Textbox(label="Password", type="password", placeholder="••••••••", lines=1)
        pwd_btn = gr.Button("Unlock", variant="primary")

    with gr.Tabs() as tabs:
        with gr.Tab("Vochi CRM"):
            with gr.Row():
                tenant_tb = gr.Textbox(label="Tenant ID", value=DEFAULT_TENANT_ID, scale=1)
                date_inp = gr.Textbox(label="Date", value=_today_str(), placeholder="YYYY-MM-DD", scale=1)
                time_from_inp = gr.Textbox(label="Time from", placeholder="HH:MM", scale=1)
                time_to_inp = gr.Textbox(label="Time to", placeholder="HH:MM", scale=1)
                call_type_dd = gr.Dropdown(choices=CALL_TYPE_OPTIONS, value="", label="Call type", type="value", scale=1)
            with gr.Row():
                filter_btn = gr.Button("Filter", variant="primary", scale=0)
                batch_btn = gr.Button("Batch analyze", variant="secondary", scale=0)
                save_btn = gr.Button("Save to file", scale=0)
            status_fetch = gr.Markdown()
            calls_df = gr.Dataframe(value=pd.DataFrame(), label="Call list", interactive=False)
            row_dd = gr.Dropdown(choices=[], label="Call", info="Choose a row to listen/analyze")
            with gr.Row():
                play_btn = gr.Button("🎧 Play")
            url_html = gr.HTML()
            audio_out = gr.Audio(label="Audio", type="filepath")
            batch_summary_md = gr.Markdown()
            batch_results_html = gr.HTML()
            batch_status_md = gr.Markdown()
            batch_file = gr.File(label="Export CSV", visible=False)

        with gr.Tab("AI Analysis"):
            with gr.Row():
                tpl_dd = gr.Dropdown(choices=TPL_OPTIONS, value="simple", label="Template")
                lang_dd = gr.Dropdown(choices=LANG_OPTIONS, value=Language.AUTO.value, label="Language")
                model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=MODEL_DEFAULT,
                    label="Model",
                    interactive=bool(MODEL_OPTIONS),
                    info=MODEL_INFO,
                )
            custom_prompt_tb = gr.Textbox(label="Custom prompt", lines=8, visible=False)
            analyze_btn = gr.Button("🧠 Analyze", variant="primary")
            analysis_md = gr.Markdown()

    filter_btn.click(
        ui_filter_calls,
        inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, authed, tenant_tb],
        outputs=[calls_df, row_dd, status_fetch, pwd_group],
    )

    pwd_btn.click(
        ui_check_password,
        inputs=[pwd_tb],
        outputs=[authed, status_fetch, pwd_group],
    )

    batch_btn.click(
        ui_mass_analyze,
        inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, tenant_tb, authed],
        outputs=[batch_results_state, batch_results_html, batch_summary_md, batch_status_md, batch_file],
    )

    play_btn.click(
        ui_play_audio,
        inputs=[row_dd, calls_df, tenant_tb],
        outputs=[url_html, audio_out, status_fetch],
    )

    tpl_dd.change(ui_toggle_custom_prompt, inputs=[tpl_dd], outputs=[custom_prompt_tb])

    analyze_btn.click(
        ui_analyze,
        inputs=[row_dd, calls_df, tpl_dd, custom_prompt_tb, lang_dd, model_dd, tenant_tb],
        outputs=[analysis_md],
    )

    save_btn.click(
        ui_export_results,
        inputs=[batch_results_state],
        outputs=[batch_file, batch_status_md],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=["D:\\tmp"])
