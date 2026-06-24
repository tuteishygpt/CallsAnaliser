"""Environment loading helpers."""
from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

from dotenv import load_dotenv


EnvPath = Union[str, PathLike[str]]


def load_environment(dotenv_path: EnvPath | None = None) -> bool:
    """Load .env files while tolerating UTF-8 BOM markers."""

    path = Path(dotenv_path) if dotenv_path is not None else None
    return load_dotenv(dotenv_path=path, encoding="utf-8-sig")
