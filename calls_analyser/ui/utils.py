"""Utility helpers for the Gradio UI."""
from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd

from .config import CALL_TYPE_OPTIONS


def label_row(row: dict) -> str:
    start = row.get("Start", "")
    src = row.get("CallerId", "")
    dst = row.get("Destination", "")
    dur = row.get("Duration", "")
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
