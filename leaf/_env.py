"""Minimal .env loader -- no python-dotenv dependency. Reads KEY=VALUE
lines from a .env file at the repo root (if present) into os.environ,
without overwriting any variable already set in the real environment.
Never logs or echoes the values it reads."""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).parent.parent / ".env"
_loaded = False


def load_dotenv_once() -> None:
    global _loaded
    if _loaded or not _ENV_PATH.exists():
        _loaded = True
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if value and key not in os.environ:
            os.environ[key] = value
    _loaded = True
