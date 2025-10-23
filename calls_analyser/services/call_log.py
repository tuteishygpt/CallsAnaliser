"""Call log service."""
from __future__ import annotations

from datetime import date
from typing import List

from calls_analyser.domain.models import CallLogEntry, RecordingHandle
from calls_analyser.ports.storage import StoragePort
from calls_analyser.ports.telephony import TelephonyPort
from calls_analyser.services.tenant import TenantConfig


class CallLogService:
    """Provides call log operations for a tenant."""

    def __init__(self, telephony: TelephonyPort, storage: StoragePort) -> None:
        self._telephony = telephony
        self._storage = storage

    def list_calls(self, day: date, tenant: TenantConfig) -> List[CallLogEntry]:
        """Return call log entries for ``day``."""

        return list(self._telephony.list_calls(day=day, tenant_id=tenant.tenant_id))

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
