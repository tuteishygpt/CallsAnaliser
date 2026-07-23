from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

import calls_analyser.services.gemini_batch as gemini_batch
from calls_analyser.services.gemini_batch import (
    BatchAnalysisResult,
    BatchTask,
    UploadedBatchInputs,
    VertexBatchRunner,
)


def _runner_for_lifecycle_test(monkeypatch, *, fail_at: str | None = None):
    runner = object.__new__(VertexBatchRunner)
    runner._bucket_name = "test-bucket"
    runner._model = "publishers/google/models/gemini-test"

    cleanup_prefixes: list[str] = []
    monkeypatch.setattr(
        runner,
        "_cleanup_gcs",
        lambda prefix: cleanup_prefixes.append(prefix),
    )

    def step(name, value):
        def invoke(*_args, **_kwargs):
            if fail_at == name:
                raise RuntimeError(name)
            return value

        return invoke

    monkeypatch.setattr(
        runner,
        "_upload_audio_to_gcs",
        step(
            "upload",
            UploadedBatchInputs(
                uri_by_key={"call-1": "gs://test-bucket/audio.mp3"},
                key_by_uri={"gs://test-bucket/audio.mp3": "call-1"},
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_jsonl_to_gcs",
        step("write", "gs://test-bucket/input.jsonl"),
    )
    monkeypatch.setattr(
        runner,
        "_poll_until_done",
        step("poll", SimpleNamespace()),
    )
    monkeypatch.setattr(
        runner,
        "_read_output_jsonl",
        step("read", {"call-1": "ok"}),
    )

    class Batches:
        def create(self, **_kwargs):
            if fail_at == "create":
                raise RuntimeError("create")
            return SimpleNamespace(name="batch-job")

    runner._client = SimpleNamespace(batches=Batches())
    return runner, cleanup_prefixes


def test_vertex_batch_runner_defaults_to_global_location(monkeypatch):
    captured = {}

    class GenAI:
        class Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

    class GCSStorage:
        class Client:
            def __init__(self, **kwargs):
                captured["gcs"] = kwargs

            def bucket(self, name):
                return SimpleNamespace(name=name)

    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setenv("GCS_BATCH_BUCKET", "test-bucket")
    monkeypatch.setattr(gemini_batch, "genai", GenAI)
    monkeypatch.setattr(gemini_batch, "gcs_storage", GCSStorage)

    VertexBatchRunner(model="models/gemini-test")

    assert captured["client"]["location"] == "global"


def test_vertex_batch_runner_uses_location_from_environment(monkeypatch):
    captured = {}

    class GenAI:
        class Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

    class GCSStorage:
        class Client:
            def __init__(self, **kwargs):
                captured["gcs"] = kwargs

            def bucket(self, name):
                return SimpleNamespace(name=name)

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GCS_BATCH_BUCKET", "test-bucket")
    monkeypatch.setattr(gemini_batch, "genai", GenAI)
    monkeypatch.setattr(gemini_batch, "gcs_storage", GCSStorage)

    VertexBatchRunner(model="models/gemini-test")

    assert captured["client"]["location"] == "global"


def test_single_batch_cleans_up_after_success(monkeypatch):
    runner, cleanup_prefixes = _runner_for_lifecycle_test(monkeypatch)

    result = runner._run_single_batch(
        [BatchTask(key="call-1", path="call.mp3", mime_type="audio/mpeg")],
        "prompt",
    )

    assert result == {"call-1": "ok"}
    assert len(cleanup_prefixes) == 1
    assert cleanup_prefixes[0].startswith("batch_")


def test_run_batch_restarts_failed_chunk_once(monkeypatch):
    runner = object.__new__(VertexBatchRunner)
    attempts = []

    def run_single_batch(tasks, prompt_text):
        attempts.append((list(tasks), prompt_text))
        if len(attempts) == 1:
            raise RuntimeError("timeout")
        return {"call-1": "ok"}

    monkeypatch.setattr(runner, "_run_single_batch", run_single_batch)

    result = runner.run_batch(
        [BatchTask(key="call-1", path="call.mp3", mime_type="audio/mpeg")],
        "prompt",
        chunk_size=25,
    )

    assert result == {"call-1": "ok"}
    assert len(attempts) == 2


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        (
            [BatchTask(key=" ", path="call.mp3", mime_type="audio/mpeg")],
            "blank task key",
        ),
        (
            [
                BatchTask(key="same", path="one.mp3", mime_type="audio/mpeg"),
                BatchTask(key="same", path="two.mp3", mime_type="audio/mpeg"),
            ],
            "duplicate task key: same",
        ),
    ],
)
def test_run_batch_results_rejects_invalid_keys_before_starting_batch(
    monkeypatch,
    tasks,
    message,
):
    runner = object.__new__(VertexBatchRunner)
    started = []
    monkeypatch.setattr(
        runner,
        "_run_single_batch",
        lambda *_args: started.append(True),
    )

    with pytest.raises(ValueError, match=message):
        runner.run_batch_results(tasks, "prompt")

    assert started == []


def test_upload_audio_preserves_both_mapping_directions(tmp_path):
    runner = object.__new__(VertexBatchRunner)
    runner._bucket_name = "bucket"
    uploaded = []

    class Blob:
        def __init__(self, name):
            self.name = name

        def upload_from_filename(self, path, content_type):
            uploaded.append((self.name, path, content_type))

    runner._gcs_bucket = SimpleNamespace(blob=lambda name: Blob(name))
    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFF")

    result = runner._upload_audio_to_gcs(
        [BatchTask(key="call-1", path=str(audio), mime_type="audio/wav")],
        "batch_x",
    )

    uri = "gs://bucket/batch_x/audio/call-1.wav"
    assert result == UploadedBatchInputs(
        uri_by_key={"call-1": uri},
        key_by_uri={uri: "call-1"},
    )
    assert uploaded == [
        ("batch_x/audio/call-1.wav", str(audio), "audio/wav"),
    ]


def _output_row(
    uri: str | None,
    *,
    text: str = "analysis",
    usage: int = 1,
    error=None,
    status=None,
):
    request = {"contents": [{"parts": []}]}
    if uri is not None:
        request["contents"][0]["parts"].append(
            {"fileData": {"fileUri": uri, "mimeType": "audio/wav"}},
        )
    row = {"request": request}
    if error is not None:
        row["error"] = error
    elif status is not None:
        row["status"] = status
    else:
        row["response"] = {
            "candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"totalTokenCount": usage},
        }
    return row


def test_read_output_correlates_reversed_rows_across_files_by_request_uri():
    runner = object.__new__(VertexBatchRunner)
    uri_1 = "gs://bucket/audio/opaque-a.wav"
    uri_2 = "gs://bucket/audio/opaque-b.wav"

    class Blob:
        def __init__(self, name, rows):
            self.name = name
            self.rows = rows

        def download_as_text(self, encoding="utf-8"):
            return "\n".join(json.dumps(row) for row in self.rows)

    runner._gcs_bucket = SimpleNamespace(
        list_blobs=lambda prefix: [
            Blob("batch_x/output/first.jsonl", [_output_row(uri_2, text="second", usage=22)]),
            Blob("batch_x/output/second.jsonl", [_output_row(uri_1, text="first", usage=11)]),
        ],
    )

    results = runner._read_output_jsonl(
        {uri_1: "call-1", uri_2: "call-2"},
        "batch_x",
    )

    assert results == {
        "call-1": BatchAnalysisResult(
            text="first",
            usage_metadata={"totalTokenCount": 11},
        ),
        "call-2": BatchAnalysisResult(
            text="second",
            usage_metadata={"totalTokenCount": 22},
        ),
    }


def test_read_output_accepts_valid_response_with_empty_status():
    runner = object.__new__(VertexBatchRunner)
    uri = "gs://bucket/audio/call.wav"

    class Blob:
        name = "batch_x/output/results.jsonl"

        @staticmethod
        def download_as_text(encoding="utf-8"):
            row = _output_row(uri, text="successful", usage=37)
            row["status"] = ""
            return json.dumps(row)

    runner._gcs_bucket = SimpleNamespace(list_blobs=lambda prefix: [Blob()])

    results = runner._read_output_jsonl({uri: "call-1"}, "batch_x")

    assert results == {
        "call-1": BatchAnalysisResult(
            text="successful",
            usage_metadata={"totalTokenCount": 37},
        ),
    }


def test_read_output_classifies_rows_without_shifting_other_results(caplog):
    runner = object.__new__(VertexBatchRunner)
    known = {
        "gs://bucket/one": "call-1",
        "gs://bucket/two": "call-2",
        "gs://bucket/three": "call-3",
        "gs://bucket/four": "call-4",
        "gs://bucket/five": "call-5",
    }

    class Blob:
        name = "batch_x/output/results.jsonl"

        @staticmethod
        def download_as_text(encoding="utf-8"):
            rows = [
                "{bad json",
                json.dumps(_output_row(None)),
                json.dumps(_output_row("gs://bucket/unknown")),
                json.dumps(_output_row("gs://bucket/one", text="first")),
                json.dumps(_output_row("gs://bucket/one", text="duplicate")),
                json.dumps(_output_row("gs://bucket/two", error={"message": "denied"})),
                json.dumps(_output_row("gs://bucket/three", status={"code": 13, "message": "internal"})),
                json.dumps(_output_row("gs://bucket/four", text="fourth")),
            ]
            return "\n".join(rows)

    runner._gcs_bucket = SimpleNamespace(list_blobs=lambda prefix: [Blob()])

    results = runner._read_output_jsonl(known, "batch_x")

    assert results["call-1"].text.startswith("Error: duplicate response")
    assert results["call-2"].text.startswith("Error:")
    assert "denied" in results["call-2"].text
    assert results["call-3"].text.startswith("Error:")
    assert "internal" in results["call-3"].text
    assert results["call-4"].text == "fourth"
    assert results["call-5"].text == "Error: missing response"
    assert "results.jsonl" in caplog.text
    assert "duplicate response" in caplog.text.lower()
    assert "line 5" in caplog.text


@pytest.mark.parametrize(
    "malformed_response",
    [
        {"candidates": None},
        {"candidates": {}},
        {"candidates": [None]},
        {"candidates": [{"content": None}]},
        {"candidates": [{"content": {"parts": None}}]},
        {"candidates": [{"content": {"parts": [None]}}]},
    ],
)
def test_malformed_known_response_becomes_item_error_and_later_row_is_parsed(
    malformed_response,
):
    runner = object.__new__(VertexBatchRunner)
    malformed_uri = "gs://bucket/malformed"
    valid_uri = "gs://bucket/valid"

    class Blob:
        name = "batch_x/output/results.jsonl"

        @staticmethod
        def download_as_text(encoding="utf-8"):
            malformed_row = _output_row(malformed_uri)
            malformed_row["response"] = malformed_response
            return "\n".join(
                [
                    json.dumps(malformed_row),
                    json.dumps(_output_row(valid_uri, text="later-valid")),
                ],
            )

    runner._gcs_bucket = SimpleNamespace(list_blobs=lambda prefix: [Blob()])

    results = runner._read_output_jsonl(
        {malformed_uri: "call-malformed", valid_uri: "call-valid"},
        "batch_x",
    )

    assert results["call-malformed"].text == "Error: malformed response payload"
    assert results["call-valid"].text == "later-valid"


def test_failed_middle_chunk_becomes_item_error_and_later_chunk_continues(monkeypatch):
    runner = object.__new__(VertexBatchRunner)
    attempts = {"call-1": 0, "call-2": 0, "call-3": 0}

    def run_single_batch(tasks, _prompt):
        key = tasks[0].key
        attempts[key] += 1
        if key == "call-2":
            raise RuntimeError("timeout")
        return {key: BatchAnalysisResult(text=f"ok-{key}")}

    monkeypatch.setattr(runner, "_run_single_batch", run_single_batch)
    tasks = [
        BatchTask(key=f"call-{number}", path=f"{number}.wav", mime_type="audio/wav")
        for number in range(1, 4)
    ]

    results = runner.run_batch_results(
        tasks,
        "prompt",
        chunk_size=1,
        max_attempts=2,
    )

    assert results == {
        "call-1": BatchAnalysisResult(text="ok-call-1"),
        "call-2": BatchAnalysisResult(text="Error: batch chunk failed: timeout"),
        "call-3": BatchAnalysisResult(text="ok-call-3"),
    }
    assert attempts == {"call-1": 1, "call-2": 2, "call-3": 1}


def test_cleanup_continues_after_one_blob_delete_fails(caplog):
    runner = object.__new__(VertexBatchRunner)
    deleted = []

    class Blob:
        def __init__(self, name, *, fails=False):
            self.name = name
            self.fails = fails

        def delete(self):
            if self.fails:
                raise RuntimeError("locked")
            deleted.append(self.name)

    runner._gcs_bucket = SimpleNamespace(
        list_blobs=lambda prefix: [
            Blob("batch_x/first", fails=True),
            Blob("batch_x/second"),
        ],
    )

    runner._cleanup_gcs("batch_x")

    assert deleted == ["batch_x/second"]
    assert "batch_x/first" in caplog.text


@pytest.mark.parametrize("fail_at", ["upload", "write", "create", "poll", "read"])
def test_single_batch_cleans_up_after_failure(monkeypatch, fail_at):
    runner, cleanup_prefixes = _runner_for_lifecycle_test(
        monkeypatch,
        fail_at=fail_at,
    )

    with pytest.raises(RuntimeError, match=fail_at):
        runner._run_single_batch(
            [BatchTask(key="call-1", path="call.mp3", mime_type="audio/mpeg")],
            "prompt",
        )

    assert len(cleanup_prefixes) == 1
    assert cleanup_prefixes[0].startswith("batch_")
