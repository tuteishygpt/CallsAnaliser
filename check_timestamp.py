#!/usr/bin/env python3
"""Check what date the timestamp represents."""

import datetime as dt

timestamp = 1761177600.0
print(f"Timestamp: {timestamp}")
print(f"UTC datetime: {dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)}")
print(f"UTC date: {dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date()}")

# Also test without timezone
print(f"Local datetime: {dt.datetime.fromtimestamp(timestamp)}")
print(f"Local date: {dt.datetime.fromtimestamp(timestamp).date()}")
