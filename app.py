"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
from typing import Optional, Dict, List, Tuple
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
        options.append((f"{provider_label} â€¢ {title}", model_key))
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
    return f"{start} | {src} â†’ {dst} ({dur}s)"


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


def _build_batch_dropdown(df: pd.DataFrame) -> gr.Update:
    """Build dropdown choices for batch results using UniqueId as value.

    Keeps selection stable when sorting by using UniqueId instead of positional index.
    """
    if df is None or df.empty:
        return gr.update(choices=[], value=None)
    opts: List[Tuple[str, str]] = []
    for _idx, row in df.iterrows():
        label = f"{row.get('Start','')} | {row.get('Caller','')} -> {row.get('Destination','')} ({row.get('Duration (s)','')}s)"
        uid = str(row.get("UniqueId", ""))
        if uid:
            opts.append((label, uid))
    value = opts[0][1] if opts else None
    return gr.update(choices=opts, value=value)


def _sort_dataframe(df: pd.DataFrame, sort_key: str, ascending: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if sort_key not in df.columns:
        return df
    try:
        # Stable sort to keep order of equal elements
        return df.sort_values(by=sort_key, ascending=ascending, kind="mergesort")
    except Exception:
        return df


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
            gr.update(value=pd.DataFrame(), visible=False),
            gr.update(choices=[], value=None),
            "ðŸ”’ Enter the password to apply the filter.",
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
        return gr.update(value=df, visible=True), dd, msg, gr.update(visible=False)
    except CallsAnalyserError as exc:
        return (
            gr.update(value=pd.DataFrame(), visible=True),
            gr.update(choices=[], value=None),
            f"Domain error: {exc}",
            gr.update(visible=False),
        )
    except Exception as exc:
        return (
            gr.update(value=pd.DataFrame(), visible=True),
            gr.update(choices=[], value=None),
            f"Load error: {exc}",
            gr.update(visible=False),
        )


def ui_play_audio(selected_idx: Optional[int], df: pd.DataFrame, tenant_id: str):
    # Allow passing UniqueId directly via dropdown value (string)
    try:
        if selected_idx is not None and not str(selected_idx).strip().lstrip('-').isdigit():
            unique_id = str(selected_idx).strip()
            if not unique_id:
                return "<em>Selected item has no UniqueId.</em>", None, ""
            try:
                tenant = tenant_service.resolve(tenant_id or None)
                handle = call_log_service.ensure_recording(unique_id, tenant)
                listen_url = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{unique_id}"
                html = f'URL: <a href="{listen_url}" target="_blank">{listen_url}</a>'
                return html, handle.local_uri, "Ready"
            except CallsAnalyserError as exc:
                return f"Playback failed: {exc}", None, ""
            except Exception as exc:
                return f"Playback failed: {exc}", None, ""
    except Exception:
        pass
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
        return html, handle.local_uri, "Ready âœ…"
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
        return "First fetch the list, choose a call, and (optionally) click â€˜ðŸŽ§ Playâ€™."
    if len(ai_registry) == 0:
        return "âŒ No AI models are configured. Add provider credentials and reload the app."
    if model_pref not in ai_registry:
        return "âŒ Selected model is not available. Check API key or provider configuration."
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
        return empty_state, pd.DataFrame(), "", "ðŸ”’ Enter the password to run batch analysis.", reset_file
    if len(ai_registry) == 0 or not BATCH_MODEL_KEY:
        return empty_state, pd.DataFrame(), "", "âŒ Batch analysis is unavailable: AI model is not configured.", reset_file
    if BATCH_MODEL_KEY not in ai_registry:
        return empty_state, pd.DataFrame(), "", "âŒ Selected model for batch analysis is unavailable.", reset_file
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
            return (empty_state, pd.DataFrame(),
                "Found: 0, processed: 0",
                "â„¹ï¸ No calls for the selected filter.",
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
                "Select": False,
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
                row_data["Status"] = "âœ…"
                success += 1
            except CallsAnalyserError as exc:
                link = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{entry.unique_id}"
                row_data["Needs follow-up"] = ""
                row_data["Reason"] = f"âŒ {exc}"
                row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>' if link else ""
                row_data["Status"] = "âŒ"
            except Exception as exc:
                link = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{entry.unique_id}"
                row_data["Needs follow-up"] = ""
                row_data["Reason"] = f"âŒ {exc}"
                row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>' if link else ""
                row_data["Status"] = "âŒ"
            rows.append(row_data)
            progress((idx + 1) / total, desc=f"Analyzing {idx + 1}/{total}")

        df = pd.DataFrame(rows)
        # Ensure boolean dtype for checkbox column
        try:
            if "Select" in df.columns:
                df["Select"] = df["Select"].astype(bool)
        except Exception:
            pass
        # Auto-select the first row for user convenience
        try:
            if not df.empty:
                df.loc[:, "Select"] = False
                df.loc[0, "Select"] = True
        except Exception:
            pass
        # Put 'Select' as the first column for easier mouse clicking
        try:
            if "Select" in df.columns:
                cols = ["Select"] + [c for c in df.columns if c != "Select"]
                df = df.loc[:, cols]
        except Exception:
            pass
        summary = f"Found: {total}, processed: {success}"
        status = "âœ… Batch analysis completed."
        return df, df, summary, status, reset_file
    except CallsAnalyserError as exc:
        return empty_state, empty_state, "", f"Analysis failed: {exc}", reset_file
    except Exception as exc:
        return empty_state, empty_state, "", f"Analysis failed: {exc}", reset_file


def ui_pick_first_batch_row(results_df: pd.DataFrame, tenant_id: str):
    """Auto-select the first row after batch analysis and update dependent UI."""
    try:
        if results_df is None or getattr(results_df, "empty", True):
            return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")
        row = results_df.iloc[0]
        row_dict: Dict[str, object] = {k: row.get(k) for k in results_df.columns}
        uid = str(row.get("UniqueId", "")).strip()
        listen_url = ""
        if uid:
            try:
                tenant = tenant_service.resolve(tenant_id or None)
                listen_url = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{uid}"
            except Exception:
                listen_url = ""
        try:
            label = f"{row_dict.get('Start','')} | {row_dict.get('Caller','')} -> {row_dict.get('Destination','')} ({row_dict.get('Duration (s)','')}s)"
        except Exception:
            label = uid or "Selected"
        dd_update = gr.update(choices=[(f"Batch: {label}", uid or "")], value=(uid or None))
        return row_dict or {}, listen_url, dd_update, (uid or ""), ui_show_current_uid(uid or "")
    except Exception:
        return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")


def ui_hide_call_list():
    """Hide the manual call list after triggering batch analysis."""

    return gr.update(visible=False)


def ui_build_batch_pick_options(results_df: pd.DataFrame):
    """Return update for batch picker dropdown from DataFrame."""
    return _build_batch_dropdown(results_df if isinstance(results_df, pd.DataFrame) else pd.DataFrame())

def ui_export_results(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        return gr.update(value=None, visible=False), "âŒ No data to export."
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        results_df.to_csv(tmp.name, index=False)
        file_path = tmp.name
    return gr.update(value=file_path, visible=True), "âœ… File is ready to save."


def ui_check_password(pwd: str):
    if not _UI_PASSWORD:
        return False, (
            "âš ï¸ <b>VOCHI_UI_PASSWORD</b> is not configured in Secrets. "
            "Add it in Settings â†’ Secrets and reload the Space."
        ), gr.update(visible=True)
    if (pwd or "").strip() == _UI_PASSWORD:
        return True, "âœ… Access granted. You can now click <b>Filter</b> and proceed.", gr.update(visible=False)
    return False, "âŒ Incorrect password. Please try again.", gr.update(visible=True)


# ----------------------------------------------------------------------------
# Password config
# ----------------------------------------------------------------------------
_UI_PASSWORD = os.environ.get("VOCHI_UI_PASSWORD", "")


# ----------------------------------------------------------------------------
# Extra handlers: batch sorting, AI bridge, chat
# ----------------------------------------------------------------------------


def ui_init_batch_controls(results_df: pd.DataFrame):
    """Initialize sort and selection controls after mass analysis."""
    if results_df is None or results_df.empty:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    sort_cols = list(results_df.columns)
    default_col = "Needs follow-up" if "Needs follow-up" in sort_cols else sort_cols[0]
    return gr.update(choices=sort_cols, value=default_col), _build_batch_dropdown(results_df)


def ui_sort_batch_results(sort_key: str, order_label: str, results_df: pd.DataFrame):
    """Sort batch results and update views and selection."""
    if results_df is None or results_df.empty:
        return results_df, results_df, gr.update(choices=[], value=None)
    ascending = (str(order_label).strip().lower() == "ascending")
    sorted_df = _sort_dataframe(results_df, sort_key, ascending)
    return sorted_df, sorted_df, _build_batch_dropdown(sorted_df)


def ui_filter_batch_results(filter_col: str, query: str, results_df: pd.DataFrame):
    """Filter batch results by substring match in selected column or all."""
    if results_df is None or results_df.empty:
        return results_df, gr.update(choices=[], value=None)
    squery = (query or "").strip()
    if not squery:
        return results_df, _build_batch_dropdown(results_df)
    try:
        if filter_col and filter_col in results_df.columns and filter_col != "*All*":
            mask = results_df[filter_col].astype(str).str.contains(squery, case=False, na=False)
        else:
            # any column contains
            mask = results_df.astype(str).apply(lambda col: col.str.contains(squery, case=False, na=False))
            mask = mask.any(axis=1)
        filtered = results_df[mask].reset_index(drop=True)
    except Exception:
        filtered = results_df
    return filtered, _build_batch_dropdown(filtered)


def ui_on_batch_select(table_value=None, results_df: pd.DataFrame | None = None, tenant_id: str = "", evt: gr.SelectData | None = None):
    """Handle DataFrame selection; return row JSON, audio URL, and set AI UID.

    Robust to different Gradio event payloads; uses stored DataFrame state.
    """
    # Prefer full row payload if Gradio provides it
    row_values = None
    try:
        if evt is not None:
            row_values = getattr(evt, "row_value", None)
            if row_values is None and isinstance(evt, dict):
                row_values = evt.get("row_value")
    except Exception:
        row_values = None
    # Resolve row index from event
    row_index = None
    try:
        if evt is not None:
            # Gradio SelectData typically has .index = (row, col)
            idx_raw = getattr(evt, "index", None)
            if isinstance(idx_raw, (list, tuple)) and idx_raw:
                row_index = int(idx_raw[0])
            elif isinstance(idx_raw, int):
                row_index = idx_raw
            # Some versions expose .row
            if row_index is None:
                row_attr = getattr(evt, "row", None)
                if isinstance(row_attr, int):
                    row_index = row_attr
    except Exception:
        row_index = None

    # Ensure we have a DataFrame when we don't have row_values
    if results_df is None and table_value is None and row_values is None:
        return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")
    if not isinstance(results_df, pd.DataFrame):
        try:
            results_df = pd.DataFrame(results_df)
        except Exception:
            results_df = pd.DataFrame()
    if row_index is None and row_values is None:
        return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")

    # Build row dict
    row_dict: Dict[str, object] = {}
    uid = ""
    try:
        # 1) Use evt.row_value when available
        if row_values is not None:
            cols = list(results_df.columns) if isinstance(results_df, pd.DataFrame) else []
            if not cols and hasattr(table_value, "columns"):
                try:
                    cols = list(getattr(table_value, "columns"))
                except Exception:
                    cols = []
            if cols and isinstance(row_values, (list, tuple)):
                n = min(len(cols), len(row_values))
                row_dict = {cols[i]: row_values[i] for i in range(n)}
            else:
                # Fallback: positional mapping
                try:
                    values_list = list(row_values)
                except Exception:
                    values_list = []
                row_dict = {str(i): v for i, v in enumerate(values_list)}
            uid = str(row_dict.get("UniqueId", "")).strip()
        # 2) Fallback to DataFrame by row index
        elif not getattr(results_df, "empty", True):
            row = results_df.iloc[int(row_index)]
            row_dict = {k: row.get(k) for k in results_df.columns}
            uid = str(row.get("UniqueId", "")).strip()
        # 3) Fallback to table value list-of-rows
        elif table_value is not None:
            try:
                values = table_value[int(row_index)] if isinstance(table_value, list) else None
                if values is not None and hasattr(results_df, "columns") and len(results_df.columns) == len(values):
                    cols = list(results_df.columns)
                    row_dict = {cols[i]: values[i] for i in range(len(cols))}
                    uid = str(row_dict.get("UniqueId", "")).strip()
            except Exception:
                pass
    except Exception:
        row_dict = row_dict or {}
        uid = uid or ""

    # Build audio listen URL using tenant config (only if UID present)
    listen_url = ""
    if uid:
        try:
            tenant = tenant_service.resolve(tenant_id or None)
            listen_url = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{uid}"
        except Exception:
            listen_url = ""

    # Prepare dropdown value to reuse existing "Call" selector
    try:
        label = f"{row_dict.get('Start','')} | {row_dict.get('Caller','')} -> {row_dict.get('Destination','')} ({row_dict.get('Duration (s)','')}s)"
    except Exception:
        label = uid
    dd_update = gr.update(choices=[(f"Batch: {label}", uid or "")], value=(uid or None))

    return row_dict or {}, listen_url, dd_update, (uid or ""), ui_show_current_uid(uid or "")


def ui_send_to_ai(selected_uid: Optional[str], current_uid: Optional[str] = ""):
    """Bridge selected batch UniqueId to AI Analysis tab via state."""
    uid = (selected_uid or current_uid or "").strip()
    if not uid:
        return "No batch item selected.", ""
    return f"Sent to AI Analysis: {uid}", uid


def ui_show_current_uid(current_uid: str):
    if not (current_uid or "").strip():
        return "No file selected for AI Analysis."
    return f"**Selected UniqueId:** `{(current_uid or '').strip()}`"


def ui_analyze_bridge(
    selected_idx: Optional[int],
    df: pd.DataFrame,
    template_key: str,
    custom_prompt: str,
    lang_code: str,
    model_pref: str,
    tenant_id: str,
    current_uid: str,
):
    """Analyze using current_uid if present, else delegate to ui_analyze."""
    uid = (current_uid or "").strip()
    if uid:
        if len(ai_registry) == 0:
            return "ï¿½?O No AI models are configured. Add provider credentials and reload the app."
        if model_pref not in ai_registry:
            return "ï¿½?O Selected model is not available. Check API key or provider configuration."
        try:
            tenant = tenant_service.resolve(tenant_id or None)
            lang = Language(lang_code)
            result = analysis_service.analyze_call(
                unique_id=uid,
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
    # Fallback to existing row-based handler
    return ui_analyze(selected_idx, df, template_key, custom_prompt, lang_code, model_pref, tenant_id)


def ui_chat_send(
    message: str,
    current_uid: str,
    template_key: str,
    custom_prompt: str,
    lang_code: str,
    model_pref: str,
    tenant_id: str,
    chat_histories: Dict[str, List[Tuple[str, str]]],
):
    uid = (current_uid or "").strip()
    if not uid:
        return [], chat_histories, "Select a file first."
    if not (message or "").strip():
        history = chat_histories.get(uid, [])
        return history, chat_histories, ""
    if len(ai_registry) == 0 or model_pref not in ai_registry:
        history = chat_histories.get(uid, [])
        return history, chat_histories, "AI model not available."
    # Build a conversational prompt that references the audio call.
    prev = chat_histories.get(uid, [])
    convo = "\n".join([f"User: {u}\nAssistant: {a}" for (u, a) in prev])
    base = custom_prompt if template_key == "custom" and custom_prompt else ""
    prompt = (
        f"You are assisting with analysis of a phone call (UniqueId: {uid}).\n"
        f"Answer the user's question based on the call audio and any prior context.\n"
        f"If the question asks to summarize or extract details, do so concisely.\n\n"
        f"Conversation so far (if any):\n{convo}\n\n"
        f"User question: {message}\n\n"
        f"Additional instructions: {base}"
    ).strip()
    try:
        tenant = tenant_service.resolve(tenant_id or None)
        lang = Language(lang_code)
        result = analysis_service.analyze_call(
            unique_id=uid,
            tenant=tenant,
            lang=lang,
            options=AnalysisOptions(
                model_key=model_pref,
                prompt_key="custom",
                custom_prompt=prompt,
            ),
        )
        assistant_text = (result.text or "").strip()
    except Exception as exc:
        assistant_text = f"Chat failed: {exc}"
    # Update history
    new_history = prev + [(message, assistant_text)]
    chat_histories[uid] = new_history
    return new_history, chat_histories, ""


def ui_chat_clear(current_uid: str, chat_histories: Dict[str, List[Tuple[str, str]]]):
    uid = (current_uid or "").strip()
    if not uid:
        return [], chat_histories
    chat_histories[uid] = []
    return [], chat_histories


# ----------------------------------------------------------------------------
# Batch fallback picker by UniqueId
# ----------------------------------------------------------------------------
def ui_on_batch_pick(uid: str, results_df: pd.DataFrame, tenant_id: str):
    """Select batch row by UniqueId (dropdown fallback)."""
    try:
        if results_df is None or getattr(results_df, "empty", True):
            return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")
        suid = (uid or "").strip()
        if not suid:
            return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")
        # locate row by UniqueId
        try:
            row = results_df.loc[results_df["UniqueId"].astype(str) == suid].iloc[0]
        except Exception:
            return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")
        row_dict: Dict[str, object] = {k: row.get(k) for k in results_df.columns}
        listen_url = ""
        try:
            tenant = tenant_service.resolve(tenant_id or None)
            listen_url = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{suid}"
        except Exception:
            listen_url = ""
        # Reuse main Call dropdown as well
        try:
            label = f"{row_dict.get('Start','')} | {row_dict.get('Caller','')} -> {row_dict.get('Destination','')} ({row_dict.get('Duration (s)','')}s)"
        except Exception:
            label = suid
        dd_update = gr.update(choices=[(f"Batch: {label}", suid)], value=suid)
        return row_dict, listen_url, dd_update, suid, ui_show_current_uid(suid)
    except Exception:
        return {}, "", gr.update(choices=[], value=None), "", ui_show_current_uid("")


def ui_on_batch_select(select_data, results_df: pd.DataFrame | None, tenant_id: str):
    """Handle direct row clicks within the batch results table."""

    base_df = results_df.copy() if isinstance(results_df, pd.DataFrame) else pd.DataFrame(results_df or {})
    if base_df is None or getattr(base_df, "empty", True):
        empty = base_df if isinstance(base_df, pd.DataFrame) else pd.DataFrame()
        return (
            empty,
            empty,
            {},
            "",
            gr.update(choices=[], value=None),
            "",
            ui_show_current_uid(""),
        )

    try:
        idx = getattr(select_data, "index", None)
        if isinstance(idx, (list, tuple)):
            row_idx = idx[0]
        else:
            row_idx = idx
        if row_idx is None:
            raise ValueError
        try:
            pos = int(row_idx)
            if pos < 0:
                raise ValueError
            target_index = base_df.index[pos]
        except Exception:
            target_index = row_idx if row_idx in base_df.index else None
        if target_index is None:
            raise ValueError
    except Exception:
        return (
            base_df,
            base_df,
            {},
            "",
            gr.update(choices=[], value=None),
            "",
            ui_show_current_uid(""),
        )

    df = base_df.copy()
    if "Select" in df.columns:
        try:
            df.loc[:, "Select"] = False
            df.loc[target_index, "Select"] = True
        except Exception:
            pass

    return ui_on_batch_toggle(df, df, tenant_id)


def ui_on_batch_toggle(table_value, results_df: pd.DataFrame | None, tenant_id: str):
    """Enforce single selection via 'Select' boolean column and update dependent UI.

    Returns: updated_df, updated_state_df, row_json, audio_url, call_dd_update, current_uid, current_uid_md
    """
    # Normalize to DataFrame
    try:
        df = table_value if isinstance(table_value, pd.DataFrame) else pd.DataFrame(table_value)
    except Exception:
        df = pd.DataFrame()

    if df is None or getattr(df, "empty", True) or "Select" not in df.columns:
        # Clear selection
        return (
            df,
            df,
            {},
            "",
            gr.update(choices=[], value=None),
            "",
            ui_show_current_uid(""),
        )

    # Find selected rows
    sel_idx = []
    try:
        sel_mask = df["Select"].astype(bool)
        sel_idx = list(df.index[sel_mask])
    except Exception:
        sel_idx = []

    # Enforce single selection: keep only the first selected row
    selected_row_idx = None
    if sel_idx:
        selected_row_idx = sel_idx[0]
        try:
            df.loc[:, "Select"] = False
            df.loc[selected_row_idx, "Select"] = True
        except Exception:
            pass
    else:
        # Nothing selected
        try:
            df.loc[:, "Select"] = False
        except Exception:
            pass
        return (
            df,
            df,
            {},
            "",
            gr.update(choices=[], value=None),
            "",
            ui_show_current_uid(""),
        )

    # Build outputs from the selected row
    try:
        row = df.loc[selected_row_idx]
        row_dict: Dict[str, object] = {k: row.get(k) for k in df.columns if k != "Select" or True}
        uid = str(row.get("UniqueId", "")).strip()
    except Exception:
        row_dict, uid = {}, ""

    listen_url = ""
    if uid:
        try:
            tenant = tenant_service.resolve(tenant_id or None)
            listen_url = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{uid}"
        except Exception:
            listen_url = ""

    try:
        label = f"{row_dict.get('Start','')} | {row_dict.get('Caller','')} -> {row_dict.get('Destination','')} ({row_dict.get('Duration (s)','')}s)"
    except Exception:
        label = uid or "Selected"
    dd_update = gr.update(choices=[(f"Batch: {label}", uid or "")], value=(uid or None))

    return df, df, (row_dict or {}), listen_url, dd_update, (uid or ""), ui_show_current_uid(uid or "")

# ----------------------------------------------------------------------------
# Build Gradio UI
# ----------------------------------------------------------------------------

def _today_str():
    return _dt.date.today().strftime("%Y-%m-%d")


with gr.Blocks(title="Vochi CRM Call Logs (Gradio)") as demo:
    gr.Markdown(
        """
        # Vochi CRM â†’ MP3 â†’ AI analysis
        *Filter calls by date, time and type, listen to recordings and run batch AI analysis.*

        """
    )

    authed = gr.State(False)
    batch_results_state = gr.State(pd.DataFrame())
    current_uid_state = gr.State("")
    chat_histories_state = gr.State({})

    with gr.Group(visible=False) as pwd_group:
        gr.Markdown("### ðŸ” Enter password")
        pwd_tb = gr.Textbox(label="Password", type="password", placeholder="â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢", lines=1)
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
            row_dd = gr.Dropdown(choices=[], label="Call", info="Choose a row to listen/analyze", type="value")
            with gr.Row():
                play_btn = gr.Button("ðŸŽ§ Play")
            url_html = gr.HTML()
            audio_out = gr.Audio(label="Audio", type="filepath")
            batch_summary_md = gr.Markdown()
            batch_results_df = gr.Dataframe(value=pd.DataFrame(), label="Batch results", interactive=True)
            # Display selection details from table clicks
            batch_selected_json = gr.JSON(label="Selected row")
            batch_audio_url_tb = gr.Textbox(label="Audio link (UniqueId)")
            # Alternative selection if click doesn't work reliably
            batch_pick_dd = gr.Dropdown(choices=[], label="Batch pick", info="Pick a row if table click fails", type="value")
            # Batch results controls removed per user request. Now use table click only.
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
            current_uid_md = gr.Markdown()
            analyze_btn = gr.Button("ðŸ§  Analyze", variant="primary")
            analysis_md = gr.Markdown()
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(label="Chat for selected file", type="tuples")
            chat_msg_tb = gr.Textbox(label="Message", lines=2)
            with gr.Row():
                chat_send_btn = gr.Button("Send", variant="secondary")
                chat_clear_btn = gr.Button("Clear chat")

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

    # Run batch and then auto-select the first row
    batch_btn.click(
        ui_mass_analyze,
        inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, tenant_tb, authed],
        outputs=[batch_results_state, batch_results_df, batch_summary_md, batch_status_md, batch_file],
    ).then(
        # Populate alternative picker dropdown
        ui_build_batch_pick_options,
        inputs=[batch_results_state],
        outputs=[batch_pick_dd],
    ).then(
        ui_pick_first_batch_row,
        inputs=[batch_results_state, tenant_tb],
        outputs=[batch_selected_json, batch_audio_url_tb, row_dd, current_uid_state, current_uid_md],
    ).then(
        ui_hide_call_list,
        inputs=[],
        outputs=[calls_df],
    )

    # Additional batch controls removed; selection is via table click.

    play_btn.click(
        ui_play_audio,
        inputs=[row_dd, calls_df, tenant_tb],
        outputs=[url_html, audio_out, status_fetch],
    )

    tpl_dd.change(ui_toggle_custom_prompt, inputs=[tpl_dd], outputs=[custom_prompt_tb])

    analyze_btn.click(
        ui_analyze_bridge,
        inputs=[row_dd, calls_df, tpl_dd, custom_prompt_tb, lang_dd, model_dd, tenant_tb, current_uid_state],
        outputs=[analysis_md],
    )

    save_btn.click(
        ui_export_results,
        inputs=[batch_results_state],
        outputs=[batch_file, batch_status_md],
    )

    # Sort/filter controls removed per request.

    # Selection via checkbox column in the table (single-select)
    batch_results_df.change(
        ui_on_batch_toggle,
        inputs=[batch_results_df, batch_results_state, tenant_tb],
        outputs=[batch_results_df, batch_results_state, batch_selected_json, batch_audio_url_tb, row_dd, current_uid_state, current_uid_md],
    )

    batch_results_df.select(
        ui_on_batch_select,
        inputs=[batch_results_state, tenant_tb],
        outputs=[batch_results_df, batch_results_state, batch_selected_json, batch_audio_url_tb, row_dd, current_uid_state, current_uid_md],
    )

    # Fallback: choose via dropdown by UniqueId
    batch_pick_dd.change(
        ui_on_batch_pick,
        inputs=[batch_pick_dd, batch_results_state, tenant_tb],
        outputs=[batch_selected_json, batch_audio_url_tb, row_dd, current_uid_state, current_uid_md],
    )

    # Direct table click selects file for AI Analysis; no button needed.

    # Chat wiring
    chat_send_btn.click(
        ui_chat_send,
        inputs=[chat_msg_tb, current_uid_state, tpl_dd, custom_prompt_tb, lang_dd, model_dd, tenant_tb, chat_histories_state],
        outputs=[chatbot, chat_histories_state, batch_status_md],
    )
    chat_clear_btn.click(
        ui_chat_clear,
        inputs=[current_uid_state, chat_histories_state],
        outputs=[chatbot, chat_histories_state],
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=["D:\\tmp"])


