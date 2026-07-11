"""MTS VATS telephony adapter."""
from __future__ import annotations

from datetime import date, datetime, time
from urllib.parse import urlparse
from typing import Iterable, List, Optional

import requests

from calls_analyser.domain.exceptions import TelephonyError
from calls_analyser.domain.models import CallLogEntry, Recording
from calls_analyser.ports.telephony import TelephonyPort


class MtsVatsTelephonyAdapter(TelephonyPort):
    """Telephony adapter for MTS VATS CRM API."""

    def __init__(self, domain: str, api_key: str, http_client: Optional[requests.Session] = None) -> None:
        self._domain = self._normalize_domain(domain)
        self._api_base = f"https://{self._domain}/crmapi/v1"
        self._api_key = api_key
        self._http = http_client or requests.Session()
        # Cache mapping from unique_id (uid) to direct recording URL from history payload.
        # This allows us to use provider-native links like those in the `record` field
        # when available, while still supporting a conventional fallback path.
        self._record_urls: dict[str, str] = {}

    def register_record_url(self, unique_id: str, url: str) -> None:
        """Remember a provider-supplied direct recording URL for a call."""

        clean_unique_id = str(unique_id or "").strip()
        clean_url = str(url or "").strip()
        if clean_unique_id and clean_url:
            self._record_urls[clean_unique_id] = clean_url

    @staticmethod
    def _normalize_domain(value: str) -> str:
        raw = (value or "").strip()
        if raw.startswith(("http://", "https://")):
            host = urlparse(raw).netloc
        else:
            host = raw.split("/")[0]
        if not host:
            raise TelephonyError(f"Invalid MTS domain value: {value!r}")
        return host

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": self._api_key,
            "Accept": "audio/*,application/json",
            "User-Agent": "calls-analyser/1.0",
        }

    def list_calls(
        self,
        day: date,
        tenant_id: str,
        time_from: Optional[time] = None,
        time_to: Optional[time] = None,
        call_type: Optional[int] = None,
    ) -> Iterable[CallLogEntry]:
        del tenant_id
        url = f"{self._api_base}/history/json"
        params: dict[str, str | int] = {
            "start": self._format_utc(day, time_from or time.min),
            "end": self._format_utc(day, time_to or time.max.replace(microsecond=0)),
            "type": self._map_call_type(call_type),
        }
        try:
            response = self._http.get(url, params=params, headers=self._headers(), timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TelephonyError(f"Failed to fetch MTS VATS call logs: {exc}") from exc

        payload = response.json()
        if not isinstance(payload, list):
            raise TelephonyError(f"Unexpected MTS VATS history payload type: {type(payload)!r}")

        entries: List[CallLogEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            unique_id = str(item.get("uid") or "")
            if not unique_id:
                continue
            # Фільтруем званкі без даступнай спасылкі на запіс:
            # калі поле `record` адсутнічае або роўна null/пустае, такі званок не вяртаем.
            record_url = item.get("record")
            if not (isinstance(record_url, str) and record_url.strip()):
                continue
            # Захоўваем прамы URL запісу для наступнага get_recording(...)
            self._record_urls[unique_id] = record_url.strip()
            started_at = self._parse_started_at(item.get("start"))
            duration = item.get("duration")
            duration_seconds = int(duration) if str(duration or "").isdigit() else None
            entries.append(
                CallLogEntry(
                    unique_id=unique_id,
                    started_at=started_at,
                    caller_id=item.get("client"),
                    destination=item.get("destination"),
                    duration_seconds=duration_seconds,
                    raw=dict(item),
                )
            )
        return entries

    @staticmethod
    def _parse_started_at(value: object) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _format_utc(day: date, value: time) -> str:
        dt = datetime.combine(day, value).replace(microsecond=0)
        return dt.strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _map_call_type(call_type: Optional[int]) -> str:
        if call_type is None:
            return "all"
        mapping = {0: "in", 1: "out"}
        return mapping.get(call_type, "all")

    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        del tenant_id
        # MTS VATS history payload exposes a direct recording URL in `record`.
        # Спачатку спрабуем выкарыстаць яго, калі ён быў атрыманы ў list_calls().
        url = self._record_urls.get(unique_id)
        if not url:
            # Калі прамы URL невядомы, выкарыстоўваем кансерватыўны fallback
            url = f"{self._api_base}/history/record/{unique_id}"
        try:
            response = self._http.get(url, headers=self._headers(), timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TelephonyError(f"Failed to fetch MTS VATS recording {unique_id}: {exc}") from exc

        return Recording(unique_id=unique_id, content=response.content, source_uri=url)
