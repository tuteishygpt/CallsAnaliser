"""Tests for UI utility functions."""
from __future__ import annotations

from typing import Optional
import datetime as _dt

import pandas as pd
import pytest

from calls_analyser.services.batch_results import EXPORT_RESULT_COLUMNS
from calls_analyser.ui.utils import label_row, prepare_results_display


def _parse_day(day_value) -> _dt.date:
    if isinstance(day_value, _dt.datetime):
        return day_value.date()
    if isinstance(day_value, _dt.date):
        return day_value
    if not day_value:
        raise ValueError("Дата не зададзена.")
    
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
        raise ValueError(f"Няправільны фармат даты: {day_value}") from exc


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
        parsed = _dt.time.fromisoformat(value)
    except ValueError as exc:
        if len(value) == 5 and value.count(":") == 1:
            parsed = _dt.time.fromisoformat(f"{value}:00")
        else:
            raise ValueError(f"Няправільны фармат часу: {value}") from exc
    return parsed.replace(microsecond=0)


class TestParseDay:
    """Test the _parse_day function."""

    def test_parse_date_object(self):
        """Test parsing a date object."""
        test_date = _dt.date(2025, 10, 24)
        result = _parse_day(test_date)
        assert result == test_date

    def test_parse_datetime_object(self):
        """Test parsing a datetime object."""
        test_datetime = _dt.datetime(2025, 10, 24, 14, 30, 0)
        result = _parse_day(test_datetime)
        # Should return date part only
        assert result == _dt.date(2025, 10, 24)
        assert isinstance(result, _dt.date)

    def test_parse_iso_string(self):
        """Test parsing an ISO format date string."""
        result = _parse_day("2025-10-24")
        assert result == _dt.date(2025, 10, 24)

    def test_parse_unix_timestamp(self):
        """Test parsing a Unix timestamp (the main fix)."""
        # Use current date as a known working timestamp
        today = _dt.date.today()
        # Convert to timestamp (seconds since epoch)
        timestamp = _dt.datetime.combine(today, _dt.time(12, 0, 0), tzinfo=_dt.timezone.utc).timestamp()
        result = _parse_day(timestamp)
        expected = today
        assert result == expected

    def test_parse_invalid_timestamp(self):
        """Test parsing an invalid timestamp."""
        with pytest.raises(ValueError, match="Няправільны фармат даты"):
            _parse_day("invalid")

    def test_parse_empty_value(self):
        """Test parsing empty/None values."""
        with pytest.raises(ValueError, match="Дата не зададзена"):
            _parse_day(None)
        with pytest.raises(ValueError, match="Дата не зададзена"):
            _parse_day("")


class TestParseTimeValue:
    """Test the _parse_time_value function."""

    def test_parse_none_empty(self):
        """Test parsing None and empty values."""
        assert _parse_time_value(None) is None
        assert _parse_time_value("") is None

    def test_parse_time_object(self):
        """Test parsing a time object."""
        test_time = _dt.time(14, 30, 0)
        result = _parse_time_value(test_time)
        assert result == test_time

    def test_parse_datetime_object(self):
        """Test parsing a datetime object."""
        test_datetime = _dt.datetime(2025, 10, 24, 14, 30, 0)
        result = _parse_time_value(test_datetime)
        assert result == _dt.time(14, 30, 0)

    def test_parse_iso_time_string(self):
        """Test parsing an ISO format time string."""
        result = _parse_time_value("14:30:00")
        assert result == _dt.time(14, 30, 0)

    def test_parse_short_time_string(self):
        """Test parsing a short time string (HH:MM)."""
        result = _parse_time_value("14:30")
        assert result == _dt.time(14, 30, 0)

    def test_parse_unix_timestamp(self):
        """Test parsing a Unix timestamp (for time)."""
        # Use current datetime as a known working timestamp
        now = _dt.datetime.now(_dt.timezone.utc)
        timestamp = now.timestamp()
        result = _parse_time_value(timestamp)
        expected = now.time().replace(microsecond=0)
        assert result == expected

    def test_parse_invalid_time(self):
        """Test parsing invalid time values."""
        with pytest.raises(ValueError, match="Няправільны фармат часу"):
            _parse_time_value("invalid")


def test_label_row_supports_vochi_api_v1_fields() -> None:
    label = label_row(
        {
            "start_time": "2026-04-22T10:00:00+03:00",
            "phone_number": "+375290000000",
            "participants": [{"extension": "150"}, {"extension": "151"}],
            "duration_seconds": 42,
        }
    )

    assert label.startswith("2026-04-22T10:00:00+03:00 | +375290000000")
    assert "150, 151" in label
    assert label.endswith("(42s)")


def test_prepare_results_display_hides_unique_id_formats_start_and_keeps_user() -> None:
    source = pd.DataFrame(
        [
            {
                "Start": "2026-06-23T17:30:10+00:00",
                "Caller": "375298708492",
                "Destination": "user",
                "Duration (s)": 16,
                "UniqueId": "7AQCK8FGTC000038",
                "user": "operator-1",
                "Needs follow-up": "Yes",
                "Reason": "Needs callback",
            }
        ]
    )

    display = prepare_results_display(source)

    assert "UniqueId" not in display.columns
    assert display.iloc[0]["Start"] == "2026-06-23 17:30:10"
    assert display.iloc[0]["user"] == "operator-1"
    assert "UniqueId" in source.columns


def test_prepare_results_display_preserves_audit_columns() -> None:
    audit_columns = [
        "Initial needs follow-up",
        "Initial reason",
        "Verification needs follow-up",
        "Verification reason",
        "Verification status",
    ]
    source = pd.DataFrame([{column: "audit" for column in audit_columns} | {"UniqueId": "call-1"}])

    display = prepare_results_display(source)

    assert list(display.columns) == audit_columns


def test_result_display_order_uses_canonical_export_schema() -> None:
    source = pd.DataFrame([{column: "" for column in reversed(EXPORT_RESULT_COLUMNS)}])

    display = prepare_results_display(source)

    assert list(display.columns) == EXPORT_RESULT_COLUMNS
