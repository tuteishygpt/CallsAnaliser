"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
from typing import Optional, List, Tuple
import json

import gradio as gr
import pandas as pd

# --- Пачатак блока, які можа патрабаваць ўстаноўкі залежнасцяў ---
try:
    from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
    from calls_analyser.adapters.secrets.env import EnvSecretsAdapter
    from calls_analyser.adapters.storage.local import LocalStorageAdapter
    from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter
    from calls_analyser.domain.exceptions import CallsAnalyserError
    from calls_analyser.domain.models import Language
    from calls_analyser.ports.ai import AIModelPort
    from calls_analyser.services.analysis import AnalysisOptions, AnalysisService
    from calls_analyser.services.call_log import CallLogService
    from calls_analyser.services.prompt import PromptService
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
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError:
    PROJECT_IMPORTS_AVAILABLE = False

    class CallsAnalyserError(Exception):
        pass

    class Language:
        RUSSIAN = "ru"
        BELARUSIAN = "be"
        ENGLISH = "en"
        AUTO = "auto"

    CFG_PROMPTS = {}
    CFG_MODEL_CANDIDATES = []
    CFG_BATCH_MODEL_KEY = ""
    CFG_BATCH_PROMPT_KEY = ""
    CFG_BATCH_PROMPT_TEXT = ""
    CFG_BATCH_LANGUAGE_CODE = "auto"
# --- Канец блока ---


PROMPTS = CFG_PROMPTS if PROJECT_IMPORTS_AVAILABLE else {}

TPL_OPTIONS = [(tpl.title, tpl.key) for tpl in PROMPTS.values()] + [("Custom", "custom")]
LANG_OPTIONS = [
    ("Russian", Language.RUSSIAN),
    ("Auto", Language.AUTO),
    ("Belarusian", Language.BELARUSIAN),
    ("English", Language.ENGLISH),
]
CALL_TYPE_OPTIONS = [
    ("All types", ""),
    ("Inbound", "0"),
    ("Outbound", "1"),
    ("Internal", "2"),
]
MODEL_CANDIDATES = CFG_MODEL_CANDIDATES if PROJECT_IMPORTS_AVAILABLE else []


# ----------------------------------------------------------------------------
# Dependency wiring
# ----------------------------------------------------------------------------
DEFAULT_TENANT_ID = os.environ.get("DEFAULT_TENANT_ID", "Amedis")
DEFAULT_BASE_URL = os.environ.get("VOCHI_BASE_URL", "https://crm.vochi.by/api")

if not PROJECT_IMPORTS_AVAILABLE:
    # заглушкі
    class MockAdapter:
        def get_optional_secret(self, _):
            return os.environ.get("GOOGLE_API_KEY")

    secrets_adapter = MockAdapter()
    storage_adapter = None
    prompt_service = None
    ai_registry = {}
    tenant_service = None
    call_log_service = None
    analysis_service = None
else:
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


def _build_model_options() -> list[tuple[str, str]]:
    """Збіраем опцыі мадэлі для выпадаючага спісу."""
    if not PROJECT_IMPORTS_AVAILABLE:
        return []
    options: list[tuple[str, str]] = []
    for title, model_key in MODEL_CANDIDATES:
        if model_key not in ai_registry:
            continue
        provider = ai_registry.get(model_key)
        provider_label = getattr(provider, "provider_name", model_key)
        options.append((f"{provider_label} • {title}", model_key))
    return options


MODEL_OPTIONS = _build_model_options()
MODEL_PLACEHOLDER_CHOICE = ("Configure GOOGLE_API_KEY to enable Gemini models", "")
MODEL_CHOICES = MODEL_OPTIONS or [MODEL_PLACEHOLDER_CHOICE]
MODEL_DEFAULT = MODEL_OPTIONS[0][1] if MODEL_OPTIONS else MODEL_PLACEHOLDER_CHOICE[1]
MODEL_INFO = (
    "Select an AI model for call analysis"
    if MODEL_OPTIONS
    else "Add GOOGLE_API_KEY to secrets and reload to enable models"
)

BATCH_PROMPT_KEY = CFG_BATCH_PROMPT_KEY
BATCH_PROMPT_TEXT = (CFG_BATCH_PROMPT_TEXT or "").strip()
BATCH_MODEL_KEY = CFG_BATCH_MODEL_KEY or MODEL_DEFAULT or ""
BATCH_LANGUAGE_CODE = CFG_BATCH_LANGUAGE_CODE
try:
    BATCH_LANGUAGE = Language(BATCH_LANGUAGE_CODE)
except ValueError:
    BATCH_LANGUAGE = Language.AUTO


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
    try:
        timestamp = float(str(day_value).strip())
        if timestamp > 1e9:
            return _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc).date()
    except (ValueError, TypeError):
        pass
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
    try:
        timestamp = float(str(time_value).strip())
        if timestamp > 1e9:
            return (
                _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc)
                .time()
                .replace(microsecond=0)
            )
    except (ValueError, TypeError):
        pass
    value = str(time_value).strip()
    if not value:
        return None
    try:
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
    try:
        return int(s)
    except ValueError:
        pass
    label_to_value = {label: v for (label, v) in CALL_TYPE_OPTIONS}
    mapped = label_to_value.get(s, "")
    try:
        return int(mapped) if mapped != "" else None
    except ValueError:
        return None


def _build_dropdown(df: pd.DataFrame):
    opts = [(_label_row(row), idx) for idx, row in df.iterrows()]
    value = opts[0][1] if opts else None
    return gr.update(choices=[(label, idx) for label, idx in opts], value=value)


def _build_batch_dropdown(df: pd.DataFrame):
    if df is None or df.empty:
        return gr.update(choices=[], value=None)
    opts: List[Tuple[str, str]] = []
    for _idx, row in df.iterrows():
        label = (
            f"{row.get('Start','')} | {row.get('Caller','')} -> "
            f"{row.get('Destination','')} ({row.get('Duration (s)','')}s)"
        )
        uid = str(row.get("UniqueId", ""))
        if uid:
            opts.append((label, uid))
    value = opts[0][1] if opts else None
    return gr.update(choices=opts, value=value)


# ----------------------------------------------------------------------------
# Gradio handlers
# ----------------------------------------------------------------------------
def ui_filter_calls(
    date_value,
    time_from_value,
    time_to_value,
    call_type_value,
    authed,
    tenant_id,
):
    """Фільтруе званкі і вяртае табліцу."""
    if not authed:
        return (
            gr.update(value=pd.DataFrame(), visible=False),
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            "🔐 Enter the password to apply the filter.",
            gr.update(visible=True),
        )

    if not PROJECT_IMPORTS_AVAILABLE:
        return (
            pd.DataFrame(),
            gr.update(visible=False),
            [],
            "Project dependencies are not loaded.",
            gr.update(visible=False),
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
        df = pd.DataFrame([entry.raw for entry in entries])
        dd = _build_dropdown(df)
        msg = f"Calls found: {len(df)}"

        return (
            gr.update(value=df, visible=True),
            gr.update(visible=False),
            dd,
            msg,
            gr.update(visible=False),
        )
    except Exception as exc:
        return (
            gr.update(value=pd.DataFrame(), visible=True),
            gr.update(visible=False),
            gr.update(choices=[], value=None),
            f"Load error: {exc}",
            gr.update(visible=False),
        )


def ui_play_audio(selected_idx, df, tenant_id):
    """Прайграць аўдыё па выбраным радку.
    Лагіка:
    - калі selected_idx выглядае як UID (не лічба) -> гуляем яго;
    - калі гэта індэкс радка -> шукаем у df і бярэм UniqueId.
    """
    if not PROJECT_IMPORTS_AVAILABLE:
        return "Project dependencies are not loaded.", None, ""

    unique_id = None

    if selected_idx is not None:
        try:
            # калі дропдаўн ужо захоўвае UID напрамую
            if not str(selected_idx).isdigit():
                unique_id = str(selected_idx)
            elif df is not None and not df.empty:
                row = df.iloc[int(selected_idx)]
                unique_id = str(row.get("UniqueId"))
        except (ValueError, IndexError):
            return "<em>Invalid selection.</em>", None, ""

    if not unique_id:
        return "<em>Select a call to play.</em>", None, ""

    try:
        tenant = tenant_service.resolve(tenant_id or None)
        handle = call_log_service.ensure_recording(unique_id, tenant)

        listen_url = (
            f"{tenant.vochi_base_url.rstrip('/')}/calllogs/"
            f"{tenant.vochi_client_id}/{unique_id}"
        )
        # ВЫПРАЎЛЕНА: Дадаем унікальны ID для JavaScript
        html = f'URL: <a id="audio-listen-link" href="{listen_url}">{listen_url}</a>'

        return html, handle.local_uri, "Ready ✅"
    except Exception as exc:
        return f"Playback failed: {exc}", None, ""


def ui_toggle_custom_prompt(template_key):
    """Паказаць/схаваць поле Custom prompt."""
    return gr.update(visible=(template_key == "custom"))


def ui_mass_analyze(
    date_value,
    time_from_value,
    time_to_value,
    call_type_value,
    tenant_id,
    authed,
):
    """
    Масавы аналіз (STREAMING).
    Гэта генератар (yield), Gradio будзе адлюстроўваць вынікі паступова.
    Паведамленні прагрэс-статусу і выніковае паведамленне ідуць буйным шрыфтам (Markdown ## / ###).
    """

    empty_df = pd.DataFrame()
    hidden_df_update = gr.update(value=empty_df, visible=False)
    hidden_file = gr.update(value=None, visible=False)

    def h3(txt: str) -> str:
        # сярэдні буйны шрыфт
        return f"### {txt}"

    def h2_success(txt: str) -> str:
        # вялікі тэкст для фінальнага выніку
        return f"## {txt}"

    def h2_error(txt: str) -> str:
        return f"## {txt}"

    # 1) праверкі доступу і канфіга
    if not authed:
        yield (
            hidden_df_update,
            h2_error("🔐 Enter the password to run batch analysis."),
            hidden_file,
        )
        return

    if not PROJECT_IMPORTS_AVAILABLE:
        yield (
            hidden_df_update,
            h2_error("Project dependencies are not loaded."),
            hidden_file,
        )
        return

    if len(ai_registry) == 0 or not BATCH_MODEL_KEY:
        yield (
            hidden_df_update,
            h2_error("❌ Batch analysis is unavailable: AI model is not configured."),
            hidden_file,
        )
        return

    # 2) асноўная логіка збору спісу званкоў
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

        if not entries:
            yield (
                hidden_df_update,
                h3("ℹ️ No calls for the selected filter."),
                hidden_file,
            )
            return

        rows = []
        total = len(entries)

        # пачатковы апдэйт
        yield (
            gr.update(value=pd.DataFrame(), visible=False),
            h3(f"Starting batch analysis for {total} call(s)..."),
            hidden_file,
        )

        # 3) цыкл аналізу
        for i, entry in enumerate(entries, start=1):
            pct = int((i / total) * 100)

            row_data = {
                "Start": entry.started_at.isoformat() if entry.started_at else "",
                "Caller": entry.caller_id or "",
                "Destination": entry.destination or "",
                "Duration (s)": entry.duration_seconds,
                "UniqueId": entry.unique_id,
            }

            try:
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

                link = (
                    f"{tenant.vochi_base_url.rstrip('/')}/calllogs/"
                    f"{tenant.vochi_client_id}/{entry.unique_id}"
                )

                # спроба structured JSON
                try:
                    text = str(result.text or "").strip()
                    l, r = text.find("{"), text.rfind("}")
                    if l != -1 and r != -1 and r > l:
                        text = text[l : r + 1]
                    payload = json.loads(text)

                    row_data["Needs follow-up"] = (
                        "Yes" if payload.get("needs_follow_up") else "No"
                    )
                    row_data["Reason"] = str(payload.get("reason") or "")
                except Exception:
                    row_data["Needs follow-up"] = ""
                    row_data["Reason"] = result.text

                # ВЫПРАЎЛЕНА: Дадаем клас для JavaScript, каб зрабіць спасылку ў новай укладцы
                row_data["Link"] = f'<a href="{link}" class="new-tab-link">Listen</a>'
                row_data["Status"] = "✅"
            except Exception as exc:
                row_data["Needs follow-up"] = ""
                row_data["Reason"] = f"❌ {exc}"
                row_data["Link"] = ""
                row_data["Status"] = "❌"

            rows.append(row_data)

            partial_df = pd.DataFrame(rows)
            interim_msg = f"Analyzing {i}/{total} ({pct}%)… UID `{entry.unique_id}`"

            # прамежкавы yield (жывое абнаўленне табліцы + статус)
            yield (
                gr.update(value=partial_df, visible=True),
                h3(interim_msg),
                hidden_file,
            )

        # 4) фінал
        final_df = pd.DataFrame(rows)
        ok_count = len(final_df[final_df["Status"] == "✅"])
        final_msg = (
            "✅ Batch analysis completed. "
            f"Found: {total}, processed successfully: {ok_count}"
        )

        yield (
            gr.update(value=final_df, visible=True),
            h2_success(final_msg),
            hidden_file,
        )

    except Exception as exc:
        yield (
            hidden_df_update,
            h2_error(f"❌ Analysis failed: {exc}"),
            hidden_file,
        )
        return


def ui_hide_call_list():
    """Схаваць ручны спіс выклікаў пасля батча, каб не блытаць карыстальніка."""
    return gr.update(visible=False)


def ui_export_results(results_df):
    """Захаваць батч-аналіз у CSV і вярнуць файл у UI."""
    if results_df is None or results_df.empty:
        return gr.update(value=None, visible=False), "❌ No data to export."

    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        results_df.to_csv(tmp.name, index=False)
        return gr.update(value=tmp.name, visible=True), "✅ File is ready to save."


def ui_check_password(pwd: str):
    """Праверка доступу ў UI."""
    _UI_PASSWORD = os.environ.get("VOCHI_UI_PASSWORD", "")

    if not _UI_PASSWORD:
        # пароль не настроены -> усім можна
        return (
            False,
            "⚠️ <b>VOCHI_UI_PASSWORD</b> is not configured. Access granted without password.",
            gr.update(visible=False),
        )

    if (pwd or "").strip() == _UI_PASSWORD:
        return True, "✅ Access granted.", gr.update(visible=False)

    return False, "❌ Incorrect password.", gr.update(visible=True)


def ui_show_current_uid(current_uid: str):
    """Паказаць выбраны UID у табе AI Analysis."""
    uid = (current_uid or "").strip()
    return (
        f"**Selected UniqueId:** `{uid}`"
        if uid
        else "No file selected for AI Analysis."
    )


def ui_analyze_bridge(
    selected_idx,
    df,
    template_key,
    custom_prompt,
    lang_code,
    model_pref,
    tenant_id,
    current_uid,
):
    """
    Аналіз адной размовы З ПРАГРЭСАМ.
    ВАЖНА:
    - Гэта цяпер генератар (yield), а не звычайная функцыя.
    - Мы не выкарыстоўваем аргумент progress=... (ён ламаецца ў Gradio 5).
    - Зрабляем некалькі крокаў:
        1) праверкі і падрыхтоўка -> yield статычны статус
        2) выклік аналізу -> пасля гэтага яшчэ адзін yield з вынікам
    - Gradio сам пакажа built-in progress bar праз show_progress="full".
    """

    # STEP 0. Вызначаем, які UID трэба аналізаваць
    uid_to_analyze = (current_uid or "").strip()
    if not uid_to_analyze and selected_idx is not None and df is not None and not df.empty:
        try:
            uid_to_analyze = str(df.iloc[int(selected_idx)].get("UniqueId") or "").strip()
        except (ValueError, IndexError):
            uid_to_analyze = ""

    # Калі няма UID -> адразу вынікаем
    if not uid_to_analyze:
        yield "Select a call from the list or batch results first."
        return

    # STEP 1. Праверкі канфігурацыі перад выклікам мадэлі
    if not PROJECT_IMPORTS_AVAILABLE:
        yield "Project dependencies are not loaded."
        return

    if len(ai_registry) == 0:
        yield "❌ No AI models are configured."
        return

    if model_pref not in ai_registry:
        yield "❌ Selected model is not available."
        return

    # паказваем карыстальніку, што пачынаем
    yield f"### Preparing analysis...\n\n- UID: `{uid_to_analyze}`\n- Model: `{model_pref}`\n- Lang: `{lang_code}`\n\nPlease wait…"

    # STEP 2. Рэальны аналіз
    try:
        tenant = tenant_service.resolve(tenant_id or None)
        lang = Language(lang_code)

        result = analysis_service.analyze_call(
            unique_id=uid_to_analyze,
            tenant=tenant,
            lang=lang,
            options=AnalysisOptions(
                model_key=model_pref,
                prompt_key=template_key,
                custom_prompt=custom_prompt,
            ),
        )

        # STEP 3. Гатова, вяртаем вынік
        yield f"### Analysis result\n\n{result.text}"

    except Exception as exc:
        yield f"Analysis failed: {exc}"


def ui_on_batch_row_select(
    displayed_df: pd.DataFrame,
    full_df_state: pd.DataFrame,
    tenant_id: str,
    evt: gr.SelectData,
):
    """
    Апрацоўвае выбар радка з табліцы вынікаў (Batch results).
    ВЫПРАЎЛЕНА: Цяпер гэтая функцыя таксама адразу абнаўляе плэер.
    """
    # Значэнні па змаўчанні, калі нешта пойдзе не так
    empty_return = (
        gr.update(choices=[], value=None), # row_dd
        "", # current_uid_state
        "No file selected for AI Analysis.", # current_uid_md
        "", # url_html
        None, # audio_out
        "Selection error.", # status_fetch
    )

    if (
        evt is None
        or displayed_df is None
        or displayed_df.empty
        or full_df_state is None
        or full_df_state.empty
    ):
        return empty_return

    try:
        visual_row_index = evt.index[0]
        clicked_row_from_view = displayed_df.iloc[visual_row_index]
        uid = str(clicked_row_from_view.get("UniqueId", "")).strip()
        if not uid:
            return empty_return

        original_row_series = full_df_state[full_df_state["UniqueId"] == uid]
        if original_row_series.empty:
            return empty_return
        original_row = original_row_series.iloc[0]
        row_dict = original_row.to_dict()

        label = (
            f"{row_dict.get('Start','')} | "
            f"{row_dict.get('Caller','')} → "
            f"{row_dict.get('Destination','')} "
            f"({row_dict.get('Duration (s)','')}s)"
        )
        dd_update = gr.update(choices=[(f"Batch: {label}", uid)], value=uid)
        uid_md_update = ui_show_current_uid(uid)

        # ВЫПРАЎЛЕНА: Дадаем логіку прайгравання аўдыё прама сюды
        try:
            tenant = tenant_service.resolve(tenant_id or None)
            handle = call_log_service.ensure_recording(uid, tenant)
            listen_url = (
                f"{tenant.vochi_base_url.rstrip('/')}/calllogs/"
                f"{tenant.vochi_client_id}/{uid}"
            )
            html = f'URL: <a id="audio-listen-link" href="{listen_url}">{listen_url}</a>'
            audio_uri = handle.local_uri
            status_msg = "Ready ✅"
        except Exception as exc:
            html, audio_uri, status_msg = f"Playback failed: {exc}", None, ""

        return dd_update, uid, uid_md_update, html, audio_uri, status_msg

    except (AttributeError, IndexError, KeyError):
        return empty_return


# ----------------------------------------------------------------------------
# Build Gradio UI
# ----------------------------------------------------------------------------
def _today_str():
    return _dt.date.today().strftime("%Y-%m-%d")


# ВЫПРАЎЛЕНА: JavaScript-код для выпраўлення спасылак
# Ён знаходзіць усе спасылкі з класам 'new-tab-link' (у табліцы) і ID 'audio-listen-link' (у плэеры)
# і дадае ім target='_blank'.
JS_FIX_LINKS = """
() => {
  setTimeout(() => {
    const links = document.querySelectorAll('a.new-tab-link, a#audio-listen-link');
    links.forEach(link => {
      if (link) {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
      }
    });
  }, 100);
}
"""


with gr.Blocks(title="Vochi CRM Call Logs (Gradio)") as demo:
    gr.Markdown(
        "# Vochi CRM → MP3 → AI analysis\n"
        "*Filter calls by date, time and type, listen to recordings and run batch AI analysis.*"
    )

    authed = gr.State(False)
    batch_results_state = gr.State(pd.DataFrame())
    current_uid_state = gr.State("")

    with gr.Group(visible=os.environ.get("VOCHI_UI_PASSWORD", "") != "") as pwd_group:
        gr.Markdown("### 🔐 Enter password")
        pwd_tb = gr.Textbox(
            label="Password", type="password", placeholder="••••••••", lines=1
        )
        pwd_btn = gr.Button("Unlock", variant="primary")

    with gr.Tabs() as tabs:
        with gr.Tab("Vochi CRM"):
            with gr.Row():
                tenant_tb = gr.Textbox(
                    label="Tenant ID", value=DEFAULT_TENANT_ID, scale=1
                )
                date_inp = gr.Textbox(
                    label="Date", value=_today_str(), placeholder="YYYY-MM-DD", scale=1
                )
                time_from_inp = gr.Textbox(
                    label="Time from", placeholder="HH:MM", scale=1
                )
                time_to_inp = gr.Textbox(label="Time to", placeholder="HH:MM", scale=1)
                call_type_dd = gr.Dropdown(
                    choices=CALL_TYPE_OPTIONS,
                    value="",
                    label="Call type",
                    type="value",
                    scale=1,
                )
            with gr.Row():
                filter_btn = gr.Button("Filter", variant="primary", scale=0)
                batch_btn = gr.Button("Batch analyze", variant="secondary", scale=0)
                save_btn = gr.Button("Save to file", scale=0)

            status_fetch = gr.Markdown()
            batch_status_md = gr.Markdown()

            calls_df = gr.DataFrame(
                value=pd.DataFrame(),
                label="Call list (manual filter)",
                interactive=False,
            )

            batch_results_df = gr.DataFrame(
                value=pd.DataFrame(),
                label="Batch results",
                interactive=True,
                visible=False,
                datatype=[
                    "str",    # Start
                    "str",    # Caller
                    "str",    # Destination
                    "number", # Duration (s)
                    "str",    # UniqueId
                    "str",    # Needs follow-up
                    "str",    # Reason
                    "markdown",  # Link
                    "str",    # Status
                ],
            )

            row_dd = gr.Dropdown(
                choices=[],
                label="Call",
                info="Choose a row to listen/analyze",
                type="value",
            )

            with gr.Row():
                play_btn = gr.Button("🎧 Play")

            url_html = gr.HTML()
            audio_out = gr.Audio(label="Audio", type="filepath")
            batch_file = gr.File(label="Export CSV", visible=False)

        with gr.Tab("AI Analysis"):
            with gr.Row():
                tpl_dd = gr.Dropdown(
                    choices=TPL_OPTIONS,
                    value="simple" if TPL_OPTIONS else "custom",
                    label="Template",
                )
                lang_dd = gr.Dropdown(
                    choices=LANG_OPTIONS,
                    value=Language.AUTO,
                    label="Language",
                )
                model_dd = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=MODEL_DEFAULT,
                    label="Model",
                    interactive=bool(MODEL_OPTIONS),
                    info=MODEL_INFO,
                )

            custom_prompt_tb = gr.Textbox(
                label="Custom prompt", lines=8, visible=False
            )

            current_uid_md = gr.Markdown(
                value="No file selected for AI Analysis."
            )

            analyze_btn = gr.Button("🧠 Analyze", variant="primary")
            analysis_md = gr.Markdown()

    # --- wiring events ---

    # пароль
    pwd_btn.click(
        ui_check_password,
        inputs=[pwd_tb],
        outputs=[authed, status_fetch, pwd_group],
    )

    # ручная фільтрацыя
    filter_btn.click(
        ui_filter_calls,
        inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, authed, tenant_tb],
        outputs=[calls_df, batch_results_df, row_dd, status_fetch, pwd_group],
    )

    # масавы аналіз
    batch_btn.click(
        fn=ui_mass_analyze,
        inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, tenant_tb, authed],
        outputs=[batch_results_df, batch_status_md, batch_file],
    ).then(
        fn=lambda df: df,
        inputs=[batch_results_df],
        outputs=[batch_results_state],
    ).then(
        fn=ui_hide_call_list,
        outputs=[calls_df],
    ).then( # ВЫПРАЎЛЕНА: запускаем JS для выпраўлення спасылак у табліцы
        fn=None, js=JS_FIX_LINKS
    )

    # ВЫПРАЎЛЕНА: выбар радка з батчу -> абнаўляем усё, уключаючы плэер
    batch_results_df.select(
        fn=ui_on_batch_row_select,
        inputs=[batch_results_df, batch_results_state, tenant_tb],
        outputs=[row_dd, current_uid_state, current_uid_md, url_html, audio_out, status_fetch],
    ).then( # ВЫПРАЎЛЕНА: запускаем JS для выпраўлення спасылкі ў плэеры
        fn=None, js=JS_FIX_LINKS
    )

    # прайграванне аўдыё
    play_btn.click(
        ui_play_audio,
        inputs=[row_dd, calls_df, tenant_tb],
        outputs=[url_html, audio_out, status_fetch],
    ).then( # ВЫПРАЎЛЕНА: запускаем JS для выпраўлення спасылкі ў плэеры
        fn=None, js=JS_FIX_LINKS
    )

    # экспарт CSV
    save_btn.click(
        ui_export_results,
        inputs=[batch_results_state],
        outputs=[batch_file, batch_status_md],
    )

    # паказаць поле для свайго prompt
    tpl_dd.change(
        ui_toggle_custom_prompt,
        inputs=[tpl_dd],
        outputs=[custom_prompt_tb],
    )

    # аналіз адной размовы з прагрэсам
    analyze_btn.click(
        fn=ui_analyze_bridge,
        inputs=[
            row_dd,
            calls_df,
            tpl_dd,
            custom_prompt_tb,
            lang_dd,
            model_dd,
            tenant_tb,
            current_uid_state,
        ],
        outputs=[analysis_md],
        show_progress="full",
    )

if __name__ == "__main__":
    # УВАГА: Замяні "D:\\tmp" на шлях, дзе ляжаць MP3-запісы,
    # каб кнопка 🎧 Play магла іх прайграваць лакальна.
    demo.launch(allowed_paths=["D:\\tmp"])
