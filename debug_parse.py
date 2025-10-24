#!/usr/bin/env python3
"""Debug script to test the _parse_day function."""

import datetime as _dt

def _parse_day(day_value) -> _dt.date:
    if isinstance(day_value, _dt.date):
        return day_value
    if isinstance(day_value, _dt.datetime):
        print(f"Input is datetime: {day_value}, type: {type(day_value)}")
        result = day_value.date()
        print(f"Result of .date(): {result}, type: {type(result)}")
        return result
    if not day_value:
        raise ValueError("Дата не зададзена.")

    # Handle Unix timestamp (float) from Gradio DateTime component
    try:
        timestamp = float(str(day_value).strip())
        # Unix timestamps are typically in seconds, check if it's reasonable
        if timestamp > 1e9:  # Unix timestamp for dates after 2001
            # Use UTC to avoid timezone issues
            result = _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc).date()
            print(f"Timestamp {timestamp} -> datetime -> date: {result}")
            return result
    except (ValueError, TypeError):
        pass

    # Try to parse as ISO format string
    try:
        return _dt.date.fromisoformat(str(day_value).strip())
    except ValueError as exc:
        raise ValueError(f"Няправільны фармат даты: {day_value}") from exc


if __name__ == "__main__":
    # Test the datetime case
    test_datetime = _dt.datetime(2025, 10, 24, 14, 30, 0)
    print(f"Testing with datetime: {test_datetime}")
    result = _parse_day(test_datetime)
    print(f"Result: {result}, type: {type(result)}")
    print(f"Expected: {_dt.date(2025, 10, 24)}, type: {type(_dt.date(2025, 10, 24))}")
    print(f"Equal? {result == _dt.date(2025, 10, 24)}")
