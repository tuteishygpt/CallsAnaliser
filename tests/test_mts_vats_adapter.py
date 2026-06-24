from __future__ import annotations

from datetime import date, time
from typing import Any

import requests

from calls_analyser.adapters.telephony.mts_vats import MtsVatsTelephonyAdapter


class FakeResponse:
    def __init__(self, json_data: Any = None, content: bytes = b"", status: int = 200) -> None:
        self._json_data = json_data
        self.content = content
        self.status_code = status

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


def test_list_calls_parses_mts_payload() -> None:
    session = FakeSession()
    session.queue(
        FakeResponse(
            json_data=[
                {
                    "uid": "abc-1",
                    "start": "20260131T101530Z",
                    "client": "375291111111",
                    "destination": "100",
                    "duration": 64,
                    "record": "https://193130978.vats.mts.by/crmapi/v1/history/record/abc-1",
                }
            ]
        )
    )
    adapter = MtsVatsTelephonyAdapter("193130978.vats.mts.by", "key", http_client=session)

    items = list(adapter.list_calls(date(2026, 1, 31), tenant_id="tenant-mts"))

    assert len(items) == 1
    assert items[0].unique_id == "abc-1"
    assert items[0].duration_seconds == 64


def test_list_calls_maps_filters_to_mts_api() -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data=[]))
    adapter = MtsVatsTelephonyAdapter("https://193130978.vats.mts.by/crmapi/v1", "key", http_client=session)

    list(
        adapter.list_calls(
            date(2026, 1, 31),
            tenant_id="tenant-mts",
            time_from=time(9, 0),
            time_to=time(10, 0),
            call_type=0,
        )
    )

    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url.endswith("/crmapi/v1/history/json")
    assert kwargs["params"]["type"] == "in"
    assert kwargs["params"]["start"] == "20260131T090000Z"
    assert kwargs["params"]["end"] == "20260131T100000Z"


def test_get_recording_downloads_audio_content() -> None:
    session = FakeSession()
    session.queue(FakeResponse(content=b"audio-bytes"))
    adapter = MtsVatsTelephonyAdapter("193130978.vats.mts.by", "key", http_client=session)

    recording = adapter.get_recording("uid-9", tenant_id="tenant-mts")

    assert recording.unique_id == "uid-9"
    assert recording.content == b"audio-bytes"
    assert recording.source_uri.endswith("/history/record/uid-9")
