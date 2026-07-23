from __future__ import annotations

from datetime import date, time
from typing import Any

import pytest
import requests

from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter
from calls_analyser.domain.exceptions import TelephonyError


class FakeResponse:
    def __init__(
        self,
        json_data: Any = None,
        content: bytes = b"",
        status: int = 200,
        headers: dict[str, str] | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self._json_data = json_data
        self._json_error = json_error
        self.content = content
        self.status_code = status
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_error:
            raise self._json_error
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[FakeResponse] = []

    def queue(self, response: FakeResponse) -> None:
        self.responses.append(response)

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def make_adapter(session: FakeSession, api_key: str = "secret-api-key") -> VochiTelephonyAdapter:
    return VochiTelephonyAdapter(
        "https://bot.example/api/v1/",
        api_key,
        http_client=session,
    )


def test_list_calls_requests_all_phone_numbers() -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data={"calls": [], "total": 0}))

    entries = list(
        make_adapter(session).list_calls(
            date(2026, 4, 22),
            tenant_id="tenant",
        )
    )

    assert entries == []
    url, kwargs = session.calls[0]
    assert url == "https://bot.example/api/v1/calls"
    assert kwargs["params"] == {
        "phone": "",
        "key": "secret-api-key",
        "date_from": "2026-04-22",
        "date_to": "2026-04-22",
        "limit": 50,
        "offset": 0,
    }


@pytest.mark.parametrize("call_type", [None, 0, 1, 2])
def test_list_calls_keeps_answered_calls_for_selected_type(
    call_type: int | None,
) -> None:
    session = FakeSession()
    session.queue(
        FakeResponse(
            json_data={
                "calls": [
                    {
                        "unique_id": f"answered-{item_type}",
                        "call_status": 2,
                        "call_type": item_type,
                    }
                    for item_type in (0, 1, 2)
                ]
                + [
                    {"unique_id": "failed-0", "call_status": 0, "call_type": 0},
                    {"unique_id": "failed-1", "call_status": 1, "call_type": 1},
                ],
                "total": 5,
            }
        )
    )

    entries = list(
        make_adapter(session).list_calls(
            date(2026, 4, 22),
            tenant_id="tenant",
            call_type=call_type,
        )
    )

    expected_types = (0, 1, 2) if call_type is None else (call_type,)
    assert [entry.unique_id for entry in entries] == [
        f"answered-{item_type}" for item_type in expected_types
    ]
    assert all(entry.raw["call_status"] == 2 for entry in entries)


def test_list_calls_paginates_and_maps_entries() -> None:
    session = FakeSession()
    first_page = [
        {
            "unique_id": f"uid-{index}",
            "phone_number": "+375290000000",
            "call_status": 2,
            "call_type": 0,
            "start_time": "2026-04-22T10:15:30+03:00",
            "duration_seconds": 12,
            "participants": [
                {"extension": "150", "status": "no_answer"},
                {"extension": "151", "status": "busy"},
            ],
            "recording_url": f"https://bot.example/recording/uid-{index}",
        }
        for index in range(50)
    ]
    session.queue(FakeResponse(json_data={"calls": first_page, "total": 52}))
    session.queue(
        FakeResponse(
            json_data={
                "calls": [
                    {
                        "unique_id": "uid-50",
                        "phone_number": "+375291111111",
                        "call_status": 2,
                        "call_type": 1,
                        "start_time": "invalid",
                        "duration_seconds": "invalid",
                        "participants": [{"extension": "152"}],
                    },
                    {
                        "unique_id": "failed",
                        "call_status": 1,
                        "call_type": 1,
                    },
                ],
                "total": 52,
            }
        )
    )

    entries = list(make_adapter(session).list_calls(date(2026, 4, 22), "tenant"))

    assert len(entries) == 51
    assert entries[0].unique_id == "uid-0"
    assert entries[0].caller_id == "+375290000000"
    assert entries[0].destination == "150, 151"
    assert entries[0].duration_seconds == 12
    assert entries[0].started_at.isoformat() == "2026-04-22T10:15:30+03:00"
    assert entries[0].raw["call_status"] == 2
    assert entries[-1].unique_id == "uid-50"
    assert entries[-1].started_at is None
    assert entries[-1].duration_seconds is None
    assert session.calls[1][1]["params"]["offset"] == 50


def test_list_calls_rejects_payload_without_calls() -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data={"total": 1}))

    with pytest.raises(TelephonyError, match="invalid calls payload"):
        list(make_adapter(session).list_calls(date(2026, 4, 22), "tenant"))


def test_list_calls_wraps_http_error_without_leaking_api_key() -> None:
    session = FakeSession()
    session.queue(FakeResponse(status=500))

    with pytest.raises(TelephonyError) as exc_info:
        list(make_adapter(session).list_calls(date(2026, 4, 22), "tenant"))

    assert "secret-api-key" not in str(exc_info.value)


def test_get_recording_fetches_metadata_then_downloads_audio() -> None:
    session = FakeSession()
    session.queue(
        FakeResponse(
            json_data={
                "recording_url": "https://bot.example/permanent/uid-9",
                "download_url": "https://s3.example/temporary-audio",
            }
        )
    )
    session.queue(
        FakeResponse(
            content=b"audio-bytes",
            headers={"Content-Type": "audio/wav"},
        )
    )

    recording = make_adapter(session).get_recording("uid-9", tenant_id="tenant")

    assert recording.unique_id == "uid-9"
    assert recording.content == b"audio-bytes"
    assert recording.content_type == "audio/wav"
    assert recording.source_uri == "https://bot.example/permanent/uid-9"
    metadata_url, metadata_kwargs = session.calls[0]
    assert metadata_url == "https://bot.example/api/v1/recording"
    assert metadata_kwargs["params"] == {
        "unique_id": "uid-9",
        "key": "secret-api-key",
    }
    download_url, download_kwargs = session.calls[1]
    assert download_url == "https://s3.example/temporary-audio"
    assert "params" not in download_kwargs


def test_get_recording_uses_registered_direct_record_url() -> None:
    session = FakeSession()
    session.queue(
        FakeResponse(
            content=b"audio-bytes",
            headers={"Content-Type": "audio/mpeg"},
        )
    )
    adapter = make_adapter(session)
    direct_url = "https://bot.example/permanent/uid-9"

    adapter.register_record_url("uid-9", direct_url)
    recording = adapter.get_recording("uid-9", tenant_id="tenant")

    assert recording.unique_id == "uid-9"
    assert recording.content == b"audio-bytes"
    assert recording.content_type == "audio/mpeg"
    assert recording.source_uri == direct_url
    assert session.calls[0][0] == direct_url
    assert "params" not in session.calls[0][1]


def test_get_recording_uses_non_secret_fallback_source_uri() -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data={"download_url": "https://s3.example/audio"}))
    session.queue(FakeResponse(content=b"audio"))

    recording = make_adapter(session).get_recording("uid-1", tenant_id="tenant")

    assert recording.source_uri == "https://bot.example/api/v1/recording/uid-1"
    assert recording.content_type == "audio/mpeg"
    assert "secret-api-key" not in recording.source_uri


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"download_url": ""},
        [],
    ],
)
def test_get_recording_rejects_missing_download_url(metadata: Any) -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data=metadata))

    with pytest.raises(TelephonyError, match="download_url"):
        make_adapter(session).get_recording("uid-1", tenant_id="tenant")


def test_get_recording_wraps_download_error_without_leaking_api_key() -> None:
    session = FakeSession()
    session.queue(FakeResponse(json_data={"download_url": "https://s3.example/audio"}))
    session.queue(FakeResponse(status=403))

    with pytest.raises(TelephonyError) as exc_info:
        make_adapter(session).get_recording("uid-1", tenant_id="tenant")

    assert "uid-1" in str(exc_info.value)
    assert "secret-api-key" not in str(exc_info.value)
