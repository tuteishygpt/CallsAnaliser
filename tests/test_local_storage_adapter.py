from __future__ import annotations

from pathlib import Path

from calls_analyser.adapters.storage.local import LocalStorageAdapter


def test_save_file_from_bytes(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=tmp_path)
    uri = adapter.save_file(b"hello", "test.bin")

    saved_path = Path(uri)
    assert saved_path.read_bytes() == b"hello"
    assert adapter.exists(uri) is True
    assert adapter.open(uri) == b"hello"


def test_save_file_from_path(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"abc")
    target_dir = tmp_path / "nested"

    adapter = LocalStorageAdapter(base_dir=target_dir)
    uri = adapter.save_file(source, "copy.bin")

    saved_path = Path(uri)
    assert saved_path.read_bytes() == b"abc"


def test_save_file_from_text(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=tmp_path)

    uri = adapter.save_file("hello", "text.txt")

    saved_path = Path(uri)
    assert saved_path.read_text(encoding="utf-8") == "hello"
