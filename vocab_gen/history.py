"""Track which deck words have actually been surfaced, and steer toward the rest.

Each generation is a fresh API call with no memory of the last one, so a mild
preference in the model — concrete, scene-building nouns are easier to work into
vivid prose than abstract ones — gets re-expressed identically every time. The
result is that a small subset of the deck keeps reappearing while most words,
including the abstract ones that most need re-exposure, are never seen again.

This module keeps a usage count per word and feeds two short lists into the
*user* message: words to prefer (rarely used) and words to avoid (just used).
That message sits outside the cached prefix, so steering costs a few dozen
tokens and leaves the prompt cache fully intact.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path

VERSION = 1
_LOCK = threading.Lock()


def default_path() -> Path:
    """XDG state dir — deliberately outside the project, so it is never committed."""
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "vocab-gen" / "usage.json"


class History:
    def __init__(self, path: Path, words: dict[str, dict] | None = None):
        self.path = path
        self.words: dict[str, dict] = words or {}

    # ---------------------------------------------------------------- io ---
    @classmethod
    def load(cls, path: Path | None = None) -> "History":
        path = Path(path or default_path())
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version") == VERSION and isinstance(raw.get("words"), dict):
                return cls(path, raw["words"])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return cls(path)  # a corrupt or missing file is not worth failing over

    def save(self) -> None:
        payload = {"version": VERSION, "words": self.words}
        with _LOCK:  # the web UI is threaded
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=1, sort_keys=True)
                os.replace(tmp, self.path)  # atomic
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    # ------------------------------------------------------------ steering ---
    def plan(
        self,
        words: list[str],
        n_prefer: int = 24,
        n_avoid: int = 12,
        rng: random.Random | None = None,
    ) -> tuple[list[str], list[str]]:
        """Return (prefer, avoid).

        `prefer` is sampled from the least-used words. Sampling matters: most of
        the deck sits at zero uses, and taking the alphabetically-first slice of
        that pool would just swap one systematic bias for another.
        """
        rng = rng or random.Random()
        seen = {k: v for k, v in self.words.items() if k in {w.lower() for w in words}}

        by_word = {w: seen.get(w.lower(), {}).get("n", 0) for w in words}
        fewest = min(by_word.values()) if by_word else 0
        pool = [w for w, n in by_word.items() if n == fewest]
        if len(pool) < n_prefer:  # top up from the next-least-used tier
            rest = sorted((w for w in words if w not in pool), key=lambda w: by_word[w])
            pool = pool + rest[: n_prefer - len(pool)]
        prefer = rng.sample(pool, min(n_prefer, len(pool)))

        recent = sorted(seen.items(), key=lambda kv: kv[1].get("last", 0), reverse=True)
        avoid = [w for w, _ in recent[:n_avoid]]
        return sorted(prefer, key=str.lower), sorted(avoid, key=str.lower)

    def record(self, used: list[str]) -> None:
        now = time.time()
        for word in used:
            entry = self.words.setdefault(word.lower(), {"n": 0, "last": 0})
            entry["n"] += 1
            entry["last"] = now

    # --------------------------------------------------------------- stats ---
    def coverage(self, words: list[str]) -> dict:
        lowered = {w.lower() for w in words}
        touched = {w for w in self.words if w in lowered}
        total_uses = sum(self.words[w]["n"] for w in touched)
        top = sorted(
            ((w, self.words[w]["n"]) for w in touched), key=lambda kv: -kv[1]
        )[:10]
        return {
            "deck": len(words),
            "seen": len(touched),
            "unseen": len(words) - len(touched),
            "uses": total_uses,
            "top": top,
        }
