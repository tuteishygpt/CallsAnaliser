"""Call log service."""
from __future__ import annotations

from datetime import date, time, datetime
from typing import List, Optional

from calls_analyser.domain.models import CallLogEntry, RecordingHandle
from calls_analyser.ports.storage import StoragePort
from calls_analyser.ports.telephony import TelephonyPort
from calls_analyser.services.tenant import TenantConfig


class CallLogService:
    """Provides call log operations for a tenant."""

    def __init__(self, telephony: TelephonyPort, storage: StoragePort) -> None:
        self._telephony = telephony
        self._storage = storage

    def list_calls(
        self,
        day: date,
        tenant: TenantConfig,
        time_from: Optional[time] = None,
        time_to: Optional[time] = None,
        call_type: Optional[int] = None,
    ) -> List[CallLogEntry]:
        """Return call log entries for ``day`` with optional filters."""
        entries = list(
            self._telephony.list_calls(
                day=day,
                tenant_id=tenant.tenant_id,
                time_from=time_from,
                time_to=time_to,
                call_type=call_type,
            )
        )

        # Client-side fallback filtering if server ignores params
        def _parse_any_datetime(value: object) -> Optional[datetime]:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return None
                # Try ISO first (supports space separator too)
                try:
                    return datetime.fromisoformat(s.replace("Z", "+00:00"))
                except ValueError:
                    pass
                # Try a few common formats
                for fmt in (
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%d.%m.%Y %H:%M:%S",
                    "%d.%m.%Y %H:%M",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y/%m/%d %H:%M",
                ):
                    try:
                        return datetime.strptime(s, fmt)
                    except ValueError:
                        continue
            return None

        def within_time_window(entry: CallLogEntry) -> bool:
            if not (time_from or time_to):
                return True
            t: Optional[time] = None
            if entry.started_at:
                t = entry.started_at.time().replace(microsecond=0)
            else:
                raw = entry.raw or {}
                # Try multiple keys for start timestamp
                for key in ("Start", "start", "StartedAt", "startedAt", "CreateDate", "Date", "Datetime", "DateTime", "StartTime"):
                    start_raw = raw.get(key)
                    dt = _parse_any_datetime(start_raw)
                    if dt is not None:
                        t = dt.time().replace(microsecond=0)
                        break
            if t is None:
                return True  # cannot determine -> don't exclude
            if time_from and t < time_from:
                return False
            if time_to and t > time_to:
                return False
            return True

        def matches_type(entry: CallLogEntry) -> bool:
            if call_type is None:
                return True
            raw = entry.raw or {}
            # Common field variants for call type
            for key in ("CallType", "callType", "calltype", "Type", "Direction"):
                v = raw.get(key)
                if v is None:
                    continue
                s = str(v).strip()
                if s.isdigit():
                    try:
                        return int(s) == call_type
                    except ValueError:
                        continue
                # Some APIs may return words; try simple mapping
                mapping = {
                    "incoming": 0,
                    "in": 0,
                    "outgoing": 1,
                    "out": 1,
                    "internal": 2,
                }
                if s.lower() in mapping:
                    return mapping[s.lower()] == call_type
            return True  # if unknown, do not exclude

        return [e for e in entries if within_time_window(e) and matches_type(e)]

    def ensure_recording(self, unique_id: str, tenant: TenantConfig) -> RecordingHandle:
        """Return a handle to a locally stored recording."""

        file_name = f"{tenant.tenant_id}_{unique_id}.mp3"
        uri = self._storage.uri_for(file_name)
        remote_uri = f"{tenant.vochi_base_url.rstrip('/')}/calllogs/{tenant.vochi_client_id}/{unique_id}"
        if not self._storage.exists(uri):
            recording = self._telephony.get_recording(unique_id=unique_id, tenant_id=tenant.tenant_id)
            uri = self._storage.save_file(recording.content, file_name)
            remote_uri = recording.source_uri or remote_uri
        return RecordingHandle(unique_id=unique_id, local_uri=uri, source_uri=remote_uri)
