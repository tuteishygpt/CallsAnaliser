from __future__ import annotations

from types import SimpleNamespace

from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
from calls_analyser.domain.models import Language
from calls_analyser.services.gemini_batch import BatchAnalysisResult, BatchTask, VertexBatchRunner


def test_gemini_adapter_stores_usage_metadata_in_result(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"RIFF")

    class Part:
        @staticmethod
        def from_bytes(**_kwargs):
            return "audio-part"

    class Client:
        class Models:
            @staticmethod
            def generate_content(**_kwargs):
                return SimpleNamespace(
                    text="analysis",
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=4115,
                        candidates_token_count=583,
                        total_token_count=4698,
                        thoughts_token_count=0,
                    ),
                )

        models = Models()

    monkeypatch.setattr("calls_analyser.adapters.ai.gemini.genai", SimpleNamespace(Client=lambda **_kwargs: Client()))
    monkeypatch.setattr("calls_analyser.adapters.ai.gemini.genai_types", SimpleNamespace(Part=Part))
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    adapter = GeminiAIAdapter(api_key="key", model="models/gemini-test", client_factory=lambda _key: Client())

    result = adapter.analyze(SimpleNamespace(path=str(audio_path)), "prompt", Language.ENGLISH)

    assert result.metadata["usage_metadata"] == {
        "promptTokenCount": 4115,
        "candidatesTokenCount": 583,
        "totalTokenCount": 4698,
        "thoughtsTokenCount": 0,
    }


def test_vertex_batch_runner_reads_usage_metadata_from_output(monkeypatch) -> None:
    runner = object.__new__(VertexBatchRunner)

    class Blob:
        name = "batch_x/output/000.jsonl"

        @staticmethod
        def download_as_text(encoding="utf-8"):
            return (
                '{"response":{"candidates":[{"content":{"parts":[{"text":"analysis"}]}}],'
                '"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,'
                '"totalTokenCount":15,"thoughtsTokenCount":0}}}\n'
            )

    runner._gcs_bucket = SimpleNamespace(list_blobs=lambda prefix: [Blob()])

    results = runner._read_output_jsonl(
        [BatchTask(key="call-1", path="call.wav", mime_type="audio/wav")],
        "batch_x",
    )

    assert results == {
        "call-1": BatchAnalysisResult(
            text="analysis",
            usage_metadata={
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "thoughtsTokenCount": 0,
            },
        )
    }


def test_run_batch_preserves_string_return_for_existing_callers(monkeypatch) -> None:
    runner = object.__new__(VertexBatchRunner)
    monkeypatch.setattr(
        runner,
        "run_batch_results",
        lambda *_args, **_kwargs: {"call-1": BatchAnalysisResult(text="analysis")},
    )

    assert runner.run_batch([BatchTask(key="call-1", path="call.wav", mime_type="audio/wav")], "prompt") == {
        "call-1": "analysis"
    }
