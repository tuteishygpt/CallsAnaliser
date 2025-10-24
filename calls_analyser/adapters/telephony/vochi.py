"""Vochi telephony adapter."""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Iterable, List, Optional

import requests

from calls_analyser.domain.exceptions import TelephonyError
from calls_analyser.domain.models import CallLogEntry, Recording
from calls_analyser.ports.telephony import TelephonyPort


class _HTTPClient:
    """Small wrapper to make requests session injectable."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def get(self, *args, **kwargs) -> requests.Response:
        return self._session.get(*args, **kwargs)


class VochiTelephonyAdapter(TelephonyPort):
    """Telephony adapter for Vochi CRM."""

    def __init__(
        self,
        base_url: str,
        client_id: str,
        bearer_token: Optional[str] = None,
        http_client: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client_id = client_id
        self._bearer = bearer_token
        self._http = _HTTPClient(http_client)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "audio/*,application/json"}
        if self._bearer:
            headers["Authorization"] = f"Bearer {self._bearer}"
        return headers

    def list_calls(
        self,
        day: date,
        tenant_id: str,
        time_from: Optional[time] = None,
        time_to: Optional[time] = None,
        call_type: Optional[int] = None,
    ) -> Iterable[CallLogEntry]:
        url = f"{self._base_url}/calllogs"
        start_value = self._format_datetime(day, time_from or time.min)
        end_value = self._format_datetime(day, time_to or time.max.replace(microsecond=0))
        params: dict[str, str | int] = {"start": start_value, "end": end_value, "clientId": self._client_id}
        if call_type is not None:
            params["calltype"] = call_type
        try:
            response = self._http.get(url, params=params, headers=self._headers(), timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TelephonyError(f"Failed to fetch call logs: {exc}") from exc

        payload = response.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        entries: List[CallLogEntry] = []
        for item in data:
            unique_id = str(item.get("UniqueId"))
            start_raw = item.get("Start")
            started_at = None
            if isinstance(start_raw, str):
                try:
                    started_at = datetime.fromisoformat(start_raw)
                except ValueError:
                    started_at = None
            entry = CallLogEntry(
                unique_id=unique_id,
                started_at=started_at,
                caller_id=item.get("CallerId"),
                destination=item.get("Destination"),
                duration_seconds=int(item["Duration"]) if str(item.get("Duration", "")).isdigit() else None,
                raw=dict(item),
            )
            entries.append(entry)
        return entries

    @staticmethod
    def _format_datetime(day: date, time_value: time) -> str:
        dt = datetime.combine(day, time_value)
        return dt.replace(microsecond=0).isoformat()

    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        url = f"{self._base_url}/calllogs/{self._client_id}/{unique_id}"
        try:
            response = self._http.get(url, headers=self._headers(), timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TelephonyError(f"Failed to fetch recording {unique_id}: {exc}") from exc
        return Recording(unique_id=unique_id, content=response.content, source_uri=url)
