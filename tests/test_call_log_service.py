from __future__ import annotations

from datetime import date
from pathlib import Path

from calls_analyser.domain.models import CallLogEntry, Recording
from calls_analyser.services.call_log import CallLogService
from calls_analyser.services.tenant import TenantConfig
from calls_analyser.adapters.storage.local import LocalStorageAdapter
from calls_analyser.ports.telephony import TelephonyPort


class FakeTelephony(TelephonyPort):
    def __init__(self) -> None:
        self.recording_calls = 0
        self.calls = [
            CallLogEntry(
                unique_id="abc",
                started_at=None,
                caller_id="100",
                destination="200",
                duration_seconds=42,
                raw={"UniqueId": "abc", "Start": "2024-06-01T10:00:00", "CallerId": "100", "Destination": "200", "Duration": 42},
            )
        ]

    def list_calls(self, day: date, tenant_id: str):
        return list(self.calls)

    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        self.recording_calls += 1
        return Recording(unique_id=unique_id, content=b"data", source_uri=f"https://example.com/{unique_id}")


def test_ensure_recording_caches_download(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    telephony = FakeTelephony()
    service = CallLogService(telephony, storage)
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api", vochi_client_id="client")

    handle1 = service.ensure_recording("abc", tenant)
    assert telephony.recording_calls == 1
    assert Path(handle1.local_uri).exists()
    assert handle1.source_uri == "https://example.com/abc"

    handle2 = service.ensure_recording("abc", tenant)
    assert telephony.recording_calls == 1
    assert handle2.local_uri == handle1.local_uri
    assert handle2.source_uri == "https://api/calllogs/client/abc"


def test_list_calls_returns_entries(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    telephony = FakeTelephony()
    service = CallLogService(telephony, storage)
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api", vochi_client_id="client")

    calls = service.list_calls(date(2024, 6, 1), tenant)
    assert len(calls) == 1
    assert calls[0].unique_id == "abc"
