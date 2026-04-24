"""Integration tests for Gemini flash-lite: sequential and Vertex AI Batch.

Requires real credentials — run manually:
    pytest tests/test_integration_gemini.py -v -s

Environment:
    GOOGLE_APPLICATION_CREDENTIALS  — path to service-account JSON
    GOOGLE_CLOUD_PROJECT            — GCP project id
    GCS_BATCH_BUCKET                — GCS bucket for batch staging
"""
from __future__ import annotations

import os
import sys
import io
import pytest
from dataclasses import dataclass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MODEL = "models/gemini-2.5-flash-lite"
PROMPT = "Апішы што ты чуеш на гэтым аўдыязапісе. Адкажы 2-3 сказамі."

AUDIO_FILES = [
    ("dummy", os.path.join(os.path.dirname(__file__), "..", "dummy.wav")),
    (
        "recording1",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "venv",
            "Lib",
            "site-packages",
            "gradio",
            "media_assets",
            "audio",
            "recording1.wav",
        ),
    ),
]

needs_creds = pytest.mark.skipif(
    not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    and not os.environ.get("GOOGLE_API_KEY"),
    reason="No GCP credentials configured",
)

needs_gcs = pytest.mark.skipif(
    not os.environ.get("GCS_BATCH_BUCKET"),
    reason="GCS_BATCH_BUCKET not set",
)


@dataclass
class SimpleAudio:
    path: str
    content: bytes = b""


@needs_creds
class TestSequentialFlashLite:
    """Sequential generate_content calls via GeminiAIAdapter."""

    def _make_adapter(self):
        from calls_analyser.adapters.ai.gemini import GeminiAIAdapter

        return GeminiAIAdapter(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            model=MODEL,
        )

    def test_two_files_sequential(self):
        from calls_analyser.domain.models import Language

        adapter = self._make_adapter()

        results = {}
        for key, path in AUDIO_FILES:
            assert os.path.isfile(path), f"Audio file not found: {path}"
            audio = SimpleAudio(path=path)
            result = adapter.analyze(
                audio=audio,
                prompt=PROMPT,
                lang=Language.RUSSIAN,
                options={"tenant_id": "integration-test"},
            )
            results[key] = result
            print(f"\n[sequential] {key}:")
            print(f"  model: {result.model}")
            print(f"  text:  {result.text[:200]}")

        assert len(results) == 2
        for key, result in results.items():
            assert result.text, f"Empty response for {key}"
            assert result.provider == "gemini"
            assert "flash-lite" in result.model


@needs_creds
@needs_gcs
class TestBatchFlashLite:
    """Vertex AI Batch API via VertexBatchRunner."""

    def test_two_files_batch(self):
        from calls_analyser.services.gemini_batch import (
            VertexBatchRunner,
            BatchTask,
            guess_mime_type,
        )

        for _, path in AUDIO_FILES:
            assert os.path.isfile(path), f"Audio file not found: {path}"

        tasks = [
            BatchTask(key=key, path=path, mime_type=guess_mime_type(path))
            for key, path in AUDIO_FILES
        ]

        runner = VertexBatchRunner(model=MODEL)
        result_map = runner.run_batch(tasks, PROMPT, chunk_size=25)

        print()
        for key, text in result_map.items():
            print(f"[batch] {key}:")
            print(f"  text: {text[:200]}")

        assert len(result_map) == 2
        for key, _ in AUDIO_FILES:
            assert key in result_map, f"Missing result for {key}"
            text = result_map[key]
            assert not text.startswith("Error:"), f"Error for {key}: {text}"
            assert len(text) > 10, f"Suspiciously short response for {key}"
