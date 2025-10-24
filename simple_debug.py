#!/usr/bin/env python3
"""Simple debug script."""

import datetime as _dt

def _parse_day(day_value) -> _dt.date:
    if isinstance(day_value, _dt.date):
        return day_value
    if isinstance(day_value, _dt.datetime):
        return day_value.date()
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

if __name__ == "__main__":
    # Test the exact same case as the failing test
    test_datetime = _dt.datetime(2025, 10, 24, 14, 30, 0)
    print(f"Input: {test_datetime}, type: {type(test_datetime)}")
    print(f"isinstance(test_datetime, _dt.datetime): {isinstance(test_datetime, _dt.datetime)}")
    print(f"isinstance(test_datetime, _dt.date): {isinstance(test_datetime, _dt.date)}")

    result = _parse_day(test_datetime)
    print(f"Result: {result}, type: {type(result)}")
    expected = _dt.date(2025, 10, 24)
    print(f"Expected: {expected}, type: {type(expected)}")
    print(f"Equal? {result == expected}")
    print(f"isinstance(result, _dt.date): {isinstance(result, _dt.date)}")
