from __future__ import annotations

from datetime import date, time
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

    def list_calls(self, day: date, tenant_id: str, time_from=None, time_to=None, call_type=None):
        return list(self.calls)

    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        self.recording_calls += 1
        return Recording(unique_id=unique_id, content=b"data", source_uri=f"https://example.com/{unique_id}")


class TenantAwareFakeTelephony(TelephonyPort):
    def __init__(self, label: str) -> None:
        self.label = label

    def list_calls(self, day: date, tenant_id: str, time_from=None, time_to=None, call_type=None):
        return [
            CallLogEntry(
                unique_id=f"{tenant_id}-{self.label}",
                started_at=None,
                caller_id=None,
                destination=None,
                duration_seconds=None,
                raw={},
            )
        ]

    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        return Recording(unique_id=unique_id, content=f"{tenant_id}:{self.label}".encode("utf-8"))


def test_ensure_recording_caches_download(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    telephony = FakeTelephony()
    service = CallLogService(telephony, storage)
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api")

    handle1 = service.ensure_recording("abc", tenant)
    assert telephony.recording_calls == 1
    assert Path(handle1.local_uri).exists()
    assert handle1.source_uri == "https://example.com/abc"

    handle2 = service.ensure_recording("abc", tenant)
    assert telephony.recording_calls == 1
    assert handle2.local_uri == handle1.local_uri
    assert handle2.source_uri == "https://api/recording/abc"


def test_ensure_recording_stores_files_under_tenant_directory(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    telephony = FakeTelephony()
    service = CallLogService(telephony, storage)
    tenant = TenantConfig(tenant_id="tenant-a", vochi_base_url="https://api")

    handle = service.ensure_recording("abc", tenant)

    assert Path(handle.local_uri) == tmp_path / "tenant-a" / "abc.mp3"
    assert Path(handle.local_uri).read_bytes() == b"data"


def test_call_log_service_resolves_telephony_per_tenant(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    service = CallLogService(
        lambda tenant: TenantAwareFakeTelephony(label=tenant.provider),
        storage,
    )

    vochi_tenant = TenantConfig(
        tenant_id="tenant-a",
        vochi_base_url="https://api-a",
        provider="vochi",
    )
    mts_tenant = TenantConfig(
        tenant_id="tenant-b",
        vochi_base_url="https://api-b",
        provider="mts_vats",
    )

    vochi_calls = service.list_calls(date(2024, 6, 1), vochi_tenant)
    mts_calls = service.list_calls(date(2024, 6, 1), mts_tenant)

    assert vochi_calls[0].unique_id == "tenant-a-vochi"
    assert mts_calls[0].unique_id == "tenant-b-mts_vats"


def test_list_calls_returns_entries(tmp_path: Path) -> None:
    storage = LocalStorageAdapter(tmp_path)
    telephony = FakeTelephony()
    service = CallLogService(telephony, storage)
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api")

    calls = service.list_calls(date(2024, 6, 1), tenant, time_from=time(8, 0), time_to=time(12, 0), call_type=0)
    assert len(calls) == 1
    assert calls[0].unique_id == "abc"
