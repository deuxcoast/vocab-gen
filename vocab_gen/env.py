"""Minimal .env loading.

Deliberately not python-dotenv: this handles the one shape this project needs
(KEY=value) and keeps the runtime dependency list at exactly one package.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_env_file(start: Path | None = None) -> Path | None:
    """First .env found walking up from `start`, else the project's own."""
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    candidate = PROJECT_ROOT / ".env"
    return candidate if candidate.is_file() else None


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Strip a trailing comment, but only on unquoted values.
            value = value.split(" #", 1)[0].strip()
        if value:  # an empty placeholder must not mask a real exported var
            values[key] = value
    return values


def load_env(start: Path | None = None) -> Path | None:
    """Load .env into os.environ without overriding what's already set.

    A real exported variable always wins over the file, which is what makes
    `ANTHROPIC_API_KEY=... vocab word` behave the way you'd expect.
    """
    path = find_env_file(start)
    if path is None:
        return None
    for key, value in parse_env(path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)
    return path
