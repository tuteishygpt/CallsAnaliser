"""Utility helpers for the Gradio UI."""
from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from calls_analyser.services.batch_results import EXPORT_RESULT_COLUMNS

from .config import CALL_TYPE_OPTIONS


BASE_RESULT_DISPLAY_COLUMNS = EXPORT_RESULT_COLUMNS


def format_start_for_display(value: object) -> str:
    """Return a compact local-readable timestamp without timezone suffix."""
    if value in (None, ""):
        return ""
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, _dt.datetime):
        return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    text = str(value).strip()
    if not text:
        return ""
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = _dt.datetime.fromisoformat(normalized)
    except ValueError:
        if "T" in text:
            return text.replace("T", " ", 1)
        return text
    return parsed.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def prepare_results_display(df: pd.DataFrame | None) -> pd.DataFrame:
    """Prepare batch results for user-facing UI/email tables."""
    if df is None or df.empty:
        return pd.DataFrame()

    prepared = df.copy()
    if "Start" in prepared.columns:
        prepared["Start"] = prepared["Start"].map(format_start_for_display)

    visible_columns = [
        column
        for column in BASE_RESULT_DISPLAY_COLUMNS
        if column in prepared.columns and column != "user"
    ]
    if "user" in prepared.columns:
        insert_at = visible_columns.index("Duration (s)") if "Duration (s)" in visible_columns else len(visible_columns)
        visible_columns.insert(insert_at, "user")

    extras = [
        column
        for column in prepared.columns
        if column not in visible_columns and column != "UniqueId"
    ]
    return prepared.loc[:, visible_columns + extras]


def label_row(row: dict) -> str:
    start = row.get("Start") or row.get("start_time") or ""
    src = row.get("CallerId") or row.get("phone_number") or ""
    dst = row.get("Destination") or ""
    if not dst:
        participants = row.get("participants")
        if isinstance(participants, list):
            dst = ", ".join(
                str(item.get("extension")).strip()
                for item in participants
                if isinstance(item, dict) and str(item.get("extension") or "").strip()
            )
    dur = row.get("Duration")
    if dur in (None, ""):
        dur = row.get("duration_seconds", "")
    return f"{start} | {src} → {dst} ({dur}s)"


def parse_day(day_value) -> _dt.date:
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


def parse_time_value(time_value) -> Optional[_dt.time]:
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


def validate_time_range(time_from: Optional[_dt.time], time_to: Optional[_dt.time]) -> None:
    if time_from and time_to and time_from > time_to:
        raise ValueError("Time 'from' must be less than or equal to time 'to'.")


def resolve_call_type(value: object) -> Optional[int]:
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


def build_dropdown(df: pd.DataFrame):
    opts = [(label_row(row), idx) for idx, row in df.iterrows()]
    value = opts[0][1] if opts else None
    return gr.update(choices=[(label, idx) for label, idx in opts], value=value)


def build_batch_dropdown(df: pd.DataFrame):
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


def today_str():
    return _dt.date.today().strftime("%Y-%m-%d")


def yesterday_str():
    return (_dt.date.today() - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
