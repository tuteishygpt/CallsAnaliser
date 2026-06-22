from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_single_batch_cleans_up_after_success(monkeypatch):
    runner, cleanup_prefixes = _runner_for_lifecycle_test(monkeypatch)

    result = runner._run_single_batch(
        [BatchTask(key="call-1", path="call.mp3", mime_type="audio/mpeg")],
        "prompt",
    )

    assert result == {"call-1": "ok"}
    assert len(cleanup_prefixes) == 1
    assert cleanup_prefixes[0].startswith("batch_")


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
