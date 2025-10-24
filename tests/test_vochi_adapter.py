from __future__ import annotations

from datetime import date, time
from typing import Any

import requests

from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter


class FakeResponse:
    def __init__(self, json_data: Any = None, content: bytes = b"", status: int = 200, text: str = "") -> None:
        self._json_data = json_data
        self.content = content
        self.status_code = status
        self.text = text

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[FakeResponse] = []

    def queue(self, response: FakeResponse) -> None:
        self.responses.append(response)

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_list_calls_parses_entries() -> None:
    session = FakeSession()
    session.queue(
        FakeResponse(
            json_data={
                "data": [
                    {"UniqueId": "1", "Start": "2024-06-01T10:00:00", "CallerId": "100", "Destination": "200", "Duration": "60"}
                ]
            }
        )
    )
    adapter = VochiTelephonyAdapter("https://api", "client", http_client=session)

    entries = list(adapter.list_calls(date(2024, 6, 1), tenant_id="tenant"))
    assert len(entries) == 1
    assert entries[0].unique_id == "1"
    assert entries[0].duration_seconds == 60


def test_list_calls_applies_optional_filters() -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data={"data": []}))
    adapter = VochiTelephonyAdapter("https://api", "client", http_client=session)

    list(adapter.list_calls(
        date(2024, 6, 1),
        tenant_id="tenant",
        time_from=time(9, 30),
        time_to=time(18, 0),
        call_type=2,
    ))

    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url.endswith("/calllogs")
    params = kwargs["params"]
    assert params["start"].startswith("2024-06-01T09:30:00")
    assert params["end"].startswith("2024-06-01T18:00:00")
    assert params["calltype"] == 2


def test_get_recording_returns_bytes() -> None:
    session = FakeSession()
    session.queue(FakeResponse(content=b"audio"))
    adapter = VochiTelephonyAdapter("https://api", "client", http_client=session)

    recording = adapter.get_recording("1", tenant_id="tenant")
    assert recording.unique_id == "1"
    assert recording.content == b"audio"
    assert recording.source_uri.endswith("/1")
