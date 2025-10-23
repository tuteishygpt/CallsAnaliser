# HF Spaces / Gradio app: Vochi CRM call logs + AI analysis
# ─────────────────────────────────────────────────────────────────────────────
# How to deploy (short):
# 1) Create a new Space (Python + Gradio).
# 2) Add a file named `app.py` with THIS code.
# 3) Add a file named `requirements.txt` with the lines below.
# 4) In the Space → Settings → Repository secrets, add:
#    - VOCHI_BASE_URL (e.g. https://crm.vochi.by/api)
#    - VOCHI_CLIENT_ID (client id string)
#    - GOOGLE_API_KEY  (API key)
#    - VOCHI_UI_PASSWORD (password to unlock the UI)

#
# UI language: English.

from __future__ import annotations
import os
import json
import datetime as _dt
from typing import List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import gradio as gr

try:
    # New Google Gemini client library
    from google import genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    genai = None
    _HAS_GENAI = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL  = os.environ.get("VOCHI_BASE_URL", "https://crm.vochi.by/api")
CLIENT_ID = os.environ.get("VOCHI_CLIENT_ID") 

# If your API needs auth, fill it here (or via VOCHI_BEARER in Secrets)
_AUTH_TOKEN = os.environ.get("VOCHI_BEARER", "").strip()
AUTH_HEADERS = {
    "Accept": "audio/*,application/json;q=0.9,*/*;q=0.8",
    **({"Authorization": f"Bearer {_AUTH_TOKEN}"} if _AUTH_TOKEN else {}),
}

# 🔒 UI password from Space Secrets (set VOCHI_UI_PASSWORD there)
_UI_PASSWORD = os.environ.get("VOCHI_UI_PASSWORD", "")

# ─────────────────────────────────────────────────────────────────────────────
# Vochi API helpers
# ─────────────────────────────────────────────────────────────────────────────
def fetch_calllogs(date_str: str):
    """Get list of calls for a given date (YYYY-MM-DD)."""
    r = requests.get(
        f"{BASE_URL}/calllogs",
        params={"start": date_str, "end": date_str, "clientId": CLIENT_ID},
        headers=AUTH_HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        return data.get("data", data)
    return data


def fetch_mp3_by_unique_id(unique_id: str) -> Tuple[str, str]:
    """Fetch call recording by UniqueId and save to /tmp. Returns (filepath, url)."""
    url = f"{BASE_URL}/calllogs/{CLIENT_ID}/{unique_id}"
    r = requests.get(url, headers=AUTH_HEADERS, timeout=120)
    r.raise_for_status()
    path = f"/tmp/call_{unique_id}.mp3"
    with open(path, "wb") as f:
        f.write(r.content)
    return path, url

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates & model options
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "simple": (
        "You are a call-center conversation analyst for a medical clinic. From the call recording, provide a brief summary:\n"
        "- Purpose of the call (appointment / results / complaint / billing / other).\n"
        "- Patient intent and expectations.\n"
        "- Outcome (booked / call-back / routed / unresolved).\n"
        "- Next steps (owner and when).\n"
        "- Patient emotion (1–5) and agent tone (1–5).\n"
        "- Alerts: urgency/risks/privacy.\n\n"
        "Keep it short (6–8 lines). End with a line: ‘Service quality rating: X/5’ and one sentence explaining the rating."
    ),
    "medium": (
        "Act as a senior service analyst. Analyze the call using this structure:\n"
        "1) Quick overview: reason for the call, intent, key facts, urgency (low/medium/high).\n"
        "2) Call flow (2–4 bullets): what was asked/answered, where friction occurred.\n"
        "3) Outcomes & tasks: concrete next actions for clinic/patient with timeframes.\n"
        "4) Emotions & empathy: patient mood; agent empathy (0–5).\n"
        "5) Procedural compliance: identity verification, disclosure of recording (if stated), no off-protocol medical advice, data accuracy.\n"
        "6) Quality rating (0–100) using rubric: greeting, verification, accuracy, empathy, issue resolution (each 0–20)."
    ),
    "detailed": (
        "You are a quality & operations analyst. Provide an in-depth analysis:\n"
        "A) Segmentation: split the call into stages with approximate timestamps (if available) and roles (Patient/Agent).\n"
        "B) Structured data for booking: full name (if stated), date of birth, phone, symptoms/complaints (list), onset/duration, possible pain level 0–10 (if mentioned), required specialist/service, preferred time windows, constraints.\n"
        "C) Triage & risks: class (routine/urgent/emergency), red flags, whether immediate escalation is needed.\n"
        "D) Compliance audit: identity/privacy checks, recording disclosure, consent to data processing, booking policies.\n"
        "E) Conversation metrics: talk ratio (agent/patient), interruptions, long pauses, notable keywords.\n"
        "F) Coaching for the agent: 3–5 concrete improvements with sample phrasing.\n\n"
        "Deliver: (1) A short patient-chart summary (2–3 sentences). (2) A task table with columns: priority, owner, due."
    ),
}

TPL_OPTIONS = [
    ("Simple", "simple"),
    ("Medium", "medium"),
    ("Detailed", "detailed"),
    ("Custom", "custom"),
]

LANG_OPTIONS = [
    ("Russian", "ru"),
    ("Auto", "default"),
    ("Belarusian", "be"),
    ("English", "en"),
]

MODEL_OPTIONS = [
    ("flash", "models/gemini-2.5-flash"),
    ("pro", "models/gemini-2.5-pro"),
    ("flash-lite", "models/gemini-2.5-flash-lite"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def label_row(row: dict) -> str:
    start = row.get("Start", "")
    src   = row.get("CallerId", "")
    dst   = row.get("Destination", "")
    dur   = row.get("Duration", "")
    return f"{start} | {src} → {dst} ({dur}s)"


def _resolve_model(client: "genai.Client", preferred: str) -> str:
    name = preferred if preferred.startswith("models/") else f"models/{preferred}"
    try:
        models = list(client.models.list())
        desired_short = name.split("/", 1)[1]
        for m in models:
            mname = getattr(m, "name", "")
            short = mname.split("/", 1)[1] if mname.startswith("models/") else mname
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if short == desired_short and ("generateContent" in methods or not methods):
                return f"models/{short}"
        # Fallback to first available
        for title, candidate in MODEL_OPTIONS:
            try:
                short = candidate.split("/", 1)[1]
                for m in models:
                    mname = getattr(m, "name", "")
                    sm = mname.split("/", 1)[1] if mname.startswith("models/") else mname
                    methods = set(getattr(m, "supported_generation_methods", []) or [])
                    if sm == short and ("generateContent" in methods or not methods):
                        return candidate
            except Exception:
                pass
    except Exception:
        pass
    return name


def _system_instruction(lang_code: str) -> str:
    if lang_code == "be":
        return "Reply in Belarusian."
    if lang_code == "ru":
        return "Reply in Russian."
    if lang_code == "en":
        return "Reply in English."
    return "Reply in the caller's language; if unclear, use concise professional English."


# ─────────────────────────────────────────────────────────────────────────────
# Gradio handlers
# ─────────────────────────────────────────────────────────────────────────────

def ui_fetch_calls(date_str: str):
    try:
        items = fetch_calllogs(date_str.strip())
        df = pd.DataFrame(items)
        opts = [(label_row(r), i) for i, r in df.iterrows()]
        msg = f"Calls found: {len(df)}"
        # Update dropdown choices and default value
        dd = gr.update(choices=[(lbl, idx) for lbl, idx in opts], value=(opts[0][1] if opts else None))
        return df, dd, msg
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:800]
        except Exception:
            pass
        return pd.DataFrame(), gr.update(choices=[], value=None), f"HTTP error: {e}\n{body}"
    except Exception as e:
        return pd.DataFrame(), gr.update(choices=[], value=None), f"Load error: {e}"


def ui_play_audio(selected_idx: Optional[int], df: pd.DataFrame):
    if selected_idx is None or df is None or df.empty:
        return "<em>First fetch the list and select a row.</em>", None, None, ""
    try:
        row = df.iloc[int(selected_idx)]
    except Exception:
        return "<em>Invalid row selection.</em>", None, None, ""
    unique_id = str(row.get("UniqueId"))
    try:
        fpath = f"/tmp/call_{unique_id}.mp3"
        url_used = f"{BASE_URL}/calllogs/{CLIENT_ID}/{unique_id}"
        # Download only if not exists (avoid re-fetch)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            fpath, url_used = fetch_mp3_by_unique_id(unique_id)
        html = f'URL: <a href="{url_used}" target="_blank">{url_used}</a>'
        return html, fpath, fpath, "Ready ✅"
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:800]
        except Exception:
            pass
        return f"HTTP error: {e}<br><pre>{body}</pre>", None, None, ""
    except Exception as e:
        return f"Playback failed: {e}", None, None, ""


def ui_toggle_custom_prompt(template_key: str):
    return gr.update(visible=(template_key == "custom"))


def ui_analyze(selected_idx: Optional[int], df: pd.DataFrame,
               template_key: str, custom_prompt: str, lang_code: str, model_pref: str):
    if df is None or df.empty or selected_idx is None:
        return "First fetch the list, choose a call, and (optionally) click ‘🎧 Play’."
    if not _HAS_GENAI:
        return "❌ google-genai library not found. Make sure it's in requirements.txt."

    try:
        row = df.iloc[int(selected_idx)]
    except Exception:
        return "Invalid row selection."

    unique_id = str(row.get("UniqueId"))
    mp3_path = f"/tmp/call_{unique_id}.mp3"

    # Ensure audio file exists (download if needed)
    try:
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            mp3_path, _ = fetch_mp3_by_unique_id(unique_id)
    except Exception as e:
        return f"Failed to obtain audio for analysis: {e}"

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return "GOOGLE_API_KEY is not set in Space Secrets. Add it in Settings → Secrets and restart the Space."

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return f"Failed to initialize the client: {e}"

    # Upload file
    try:
        uploaded_file = client.files.upload(file=mp3_path)
    except Exception as e:
        return f"File upload error: {e}"

    # Prepare prompt
    if template_key == "custom":
        prompt = (custom_prompt or "").strip() or PROMPT_TEMPLATES["simple"]
    else:
        prompt = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["simple"])

    sys_inst = _system_instruction(lang_code)
    model_name = _resolve_model(client, model_pref)

    # Call model
    try:
        merged = f"""[SYSTEM INSTRUCTION: {sys_inst}]

{prompt}"""
        resp = client.models.generate_content(model=model_name, contents=[uploaded_file, merged])
        text = getattr(resp, "text", None)
        if not text:
            return "Analysis finished but returned no text. Check model settings and file format."
        return f"### Analysis result\n\n{text}"
    except Exception as e:
        # Try to attach more error details
        msg = str(e)
        try:
            if hasattr(e, "args") and e.args:
                msg = msg + "\n\n" + str(e.args[0])
        except Exception:
            pass
        return f"Error during model call: {msg}"
    finally:
        # Best-effort cleanup of remote file
        try:
            if 'uploaded_file' in locals() and hasattr(uploaded_file, 'name'):
                client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Password / gating helpers
# ─────────────────────────────────────────────────────────────────────────────

def ui_check_password(pwd: str):
    """
    Check password against VOCHI_UI_PASSWORD.
    Returns: (authed_state, status_msg_md, pwd_group_visibility)
    """
    if not _UI_PASSWORD:
        # Admin hint if password not configured
        return False, (
            "⚠️ <b>VOCHI_UI_PASSWORD</b> не наладжаны ў Secrets. "
            "Дадайце яго ў Settings → Secrets і перазапусціце Space."
        ), gr.update(visible=True)

    if (pwd or "").strip() == _UI_PASSWORD:
        return True, "✅ Доступ адкрыты. Цяпер можна націскаць <b>Fetch list</b> і працаваць.", gr.update(visible=False)
    else:
        return False, "❌ Няправільны пароль. Паспрабуйце яшчэ раз.", gr.update(visible=True)


def ui_fetch_or_auth(date_str: str, authed: bool):
    """
    If not authed, open password box instead of fetching.
    Otherwise, fetch calls.
    Returns: calls_df, row_dd, status_md, pwd_group_visibility
    """
    if not authed:
        return gr.update(), gr.update(), "🔒 Увядзіце пароль, каб атрымаць званкі.", gr.update(visible=True)
    df, dd, msg = ui_fetch_calls(date_str)
    return df, dd, msg, gr.update(visible=False)


# ─────────────────────────────────────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def _today_str():
    return _dt.date.today().strftime("%Y-%m-%d")

with gr.Blocks(title="Vochi CRM Call Logs (Gradio)") as demo:
    gr.Markdown(
        """
        # Vochi CRM → MP3 → AI analysis
        *Fetch daily calls, play/download MP3, and analyze the call with an AI model.*
        
        """
    )

    # Auth state (False by default)
    authed = gr.State(False)

    # Password "modal" (group shown on demand)
    with gr.Group(visible=False) as pwd_group:
        gr.Markdown("### 🔐 Увядзіце пароль")
        pwd_tb = gr.Textbox(label="Password", type="password", placeholder="••••••••", lines=1)
        pwd_btn = gr.Button("Адкрыць доступ", variant="primary")

    with gr.Tabs() as tabs:
        with gr.Tab("Vochi CRM"):
            with gr.Row():
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
                lang_dd = gr.Dropdown(choices=LANG_OPTIONS, value="default", label="Language")
                model_dd = gr.Dropdown(choices=MODEL_OPTIONS, value="models/gemini-2.5-flash", label="Model")
            custom_prompt_tb = gr.Textbox(label="Custom prompt", lines=8, visible=False)
            analyze_btn = gr.Button("🧠 Analyze", variant="primary")
            analysis_md = gr.Markdown()

    # Wire events
    # 1) Fetch button: gate by password
    fetch_btn.click(
        ui_fetch_or_auth,
        inputs=[date_inp, authed],
        outputs=[calls_df, row_dd, status_fetch, pwd_group],
    )

    # 2) Password submit → set authed state, show message, hide group on success
    pwd_btn.click(
        ui_check_password,
        inputs=[pwd_tb],
        outputs=[authed, status_fetch, pwd_group],
    )

    # 3) Other interactions
    play_btn.click(ui_play_audio, inputs=[row_dd, calls_df], outputs=[url_html, audio_out, file_out, status_fetch])
    tpl_dd.change(ui_toggle_custom_prompt, inputs=[tpl_dd], outputs=[custom_prompt_tb])
    analyze_btn.click(
        ui_analyze,
        inputs=[row_dd, calls_df, tpl_dd, custom_prompt_tb, lang_dd, model_dd],
        outputs=[analysis_md],
    )


if __name__ == "__main__":
    # On HF Spaces, just running this file is enough; launch() is fine for local dev, too.
    demo.launch()
