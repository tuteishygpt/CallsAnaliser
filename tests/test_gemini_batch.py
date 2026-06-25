from __future__ import annotations

from types import SimpleNamespace

import pytest

import calls_analyser.services.gemini_batch as gemini_batch
from calls_analyser.services.gemini_batch import BatchTask, VertexBatchRunner


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
        step("upload", {"call-1": "gs://test-bucket/audio.mp3"}),
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
