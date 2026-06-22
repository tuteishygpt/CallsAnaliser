"""Vochi API v1 telephony adapter."""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Iterable, Optional

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
    """Telephony adapter for the VoChi bot API."""

    _PAGE_SIZE = 50

    def __init__(
        self,
        base_url: str,
        api_key: str,
        http_client: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = _HTTPClient(http_client)

    @staticmethod
    def _headers() -> dict[str, str]:
        return {"Accept": "application/json"}

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _parse_duration(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _participant_extensions(value: Any) -> Optional[str]:
        if not isinstance(value, list):
            return None
        extensions = [
            str(item.get("extension")).strip()
            for item in value
            if isinstance(item, dict) and str(item.get("extension") or "").strip()
        ]
        return ", ".join(extensions) or None

    def list_calls(
        self,
        day: date,
        tenant_id: str,
        time_from: Optional[time] = None,
        time_to: Optional[time] = None,
        call_type: Optional[int] = None,
    ) -> Iterable[CallLogEntry]:
        del tenant_id, time_from, time_to

        url = f"{self._base_url}/calls"
        offset = 0
        entries: list[CallLogEntry] = []

        while True:
            params: dict[str, str | int] = {
                "phone": "",
                "key": self._api_key,
                "date_from": day.isoformat(),
                "date_to": day.isoformat(),
                "limit": self._PAGE_SIZE,
                "offset": offset,
            }
            try:
                response = self._http.get(
                    url,
                    params=params,
                    headers=self._headers(),
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise TelephonyError("Failed to fetch VoChi calls") from exc

            if not isinstance(payload, dict) or not isinstance(payload.get("calls"), list):
                raise TelephonyError("VoChi returned an invalid calls payload")

            calls = payload["calls"]
            for item in calls:
                if not isinstance(item, dict):
                    continue
                unique_id = str(item.get("unique_id") or "").strip()
                if not unique_id:
                    continue
                item_status = self._parse_duration(item.get("call_status"))
                item_type = self._parse_duration(item.get("call_type"))
                if item_status != 2:
                    continue
                if call_type is not None and item_type != call_type:
                    continue
                entries.append(
                    CallLogEntry(
                        unique_id=unique_id,
                        started_at=self._parse_datetime(item.get("start_time")),
                        caller_id=(
                            str(item["phone_number"])
                            if item.get("phone_number") is not None
                            else None
                        ),
                        destination=self._participant_extensions(item.get("participants")),
                        duration_seconds=self._parse_duration(item.get("duration_seconds")),
                        raw=dict(item),
                    )
                )

            total = self._parse_duration(payload.get("total"))
            page_count = len(calls)
            if (
                page_count == 0
                or page_count < self._PAGE_SIZE
                or (total is not None and offset + page_count >= total)
            ):
                break
            offset += page_count

        return entries

    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        del tenant_id

        metadata_url = f"{self._base_url}/recording"
        try:
            metadata_response = self._http.get(
                metadata_url,
                params={"unique_id": unique_id, "key": self._api_key},
                headers=self._headers(),
                timeout=60,
            )
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
        except (requests.RequestException, ValueError) as exc:
            raise TelephonyError(
                f"Failed to fetch VoChi recording metadata for {unique_id}"
            ) from exc

        if not isinstance(metadata, dict):
            raise TelephonyError(
                f"VoChi recording metadata for {unique_id} has no download_url"
            )
        download_url = str(metadata.get("download_url") or "").strip()
        if not download_url:
            raise TelephonyError(
                f"VoChi recording metadata for {unique_id} has no download_url"
            )

        try:
            recording_response = self._http.get(
                download_url,
                headers={"Accept": "audio/*"},
                timeout=120,
            )
            recording_response.raise_for_status()
        except requests.RequestException as exc:
            raise TelephonyError(
                f"Failed to download VoChi recording {unique_id}"
            ) from exc

        permanent_url = str(metadata.get("recording_url") or "").strip()
        source_uri = permanent_url or f"{metadata_url}/{unique_id}"
        content_type = (
            str(recording_response.headers.get("Content-Type") or "audio/mpeg")
            .split(";", 1)[0]
            .strip()
        )
        return Recording(
            unique_id=unique_id,
            content=recording_response.content,
            content_type=content_type,
            source_uri=source_uri,
        )
