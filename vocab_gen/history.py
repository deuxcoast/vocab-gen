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

from .morphology import same_term

VERSION = 1
KEEP_LIMIT = 40  # how many chosen sentences to retain
_LOCK = threading.Lock()


def default_path() -> Path:
    """XDG state dir — deliberately outside the project, so it is never committed."""
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "vocab-gen" / "usage.json"


class History:
    def __init__(
        self,
        path: Path,
        words: dict[str, dict] | None = None,
        kept: list[dict] | None = None,
    ):
        self.path = path
        self.words: dict[str, dict] = words or {}
        self.kept: list[dict] = kept or []

    # ---------------------------------------------------------------- io ---
    @classmethod
    def load(cls, path: Path | None = None) -> "History":
        path = Path(path or default_path())
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version") == VERSION and isinstance(raw.get("words"), dict):
                kept = raw.get("kept")
                return cls(path, raw["words"], kept if isinstance(kept, list) else [])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return cls(path)  # a corrupt or missing file is not worth failing over

    def save(self) -> None:
        payload = {"version": VERSION, "words": self.words, "kept": self.kept[-KEEP_LIMIT:]}
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
    WEIGHTINGS = ("fsrs", "lapses", "uniform")

    def plan(
        self,
        vocab,
        n_prefer: int = 24,
        n_avoid: int = 12,
        rng: random.Random | None = None,
        weighting: str = "fsrs",
        target: str | None = None,
    ):
        """Return (prefer, avoid).

        `weighting` chooses how the sample is biased: by FSRS retrievability,
        by the hand-rolled score it replaced, or not at all. The last is the
        control — if FSRS and uniform behave alike, the weighting is doing
        nothing.

        `target` is the word being taught, and is held out of both lists. A deck
        can contain the word you are making a new card for — the golden set alone
        has four — and offering it back with the learner's own definition attached
        hands over the answer. Held out inflection-aware, so a deck entry of
        "flagons" is excluded when the target is "flagon".

        `prefer` is drawn from the least-surfaced words, then sampled *weighted
        by forgetting probability* — so a word you keep lapsing on is likelier to come up
        than one you have never missed. Sampling rather than ranking still
        matters: most of the deck sits at zero uses, and taking a deterministic
        slice would trade one systematic bias for another.
        """
        rng = rng or random.Random()
        by_term = {
            w.term: w
            for w in vocab
            if not (target and same_term(w.term, target))
        }
        lowered = {t.lower() for t in by_term}
        seen = {k: v for k, v in self.words.items() if k in lowered}

        uses = {t: seen.get(t.lower(), {}).get("n", 0) for t in by_term}
        fewest = min(uses.values()) if uses else 0
        pool = [t for t, n in uses.items() if n == fewest]
        if len(pool) < n_prefer:
            rest = sorted((t for t in by_term if t not in pool), key=lambda t: uses[t])
            pool += rest[: n_prefer - len(pool)]

        if weighting == "uniform":
            weights = [1.0 for _ in pool]
        elif weighting == "lapses":
            weights = [by_term[t].legacy_shakiness for t in pool]
        else:
            weights = [by_term[t].shakiness for t in pool]
        prefer_terms = _weighted_sample(
            pool, weights, min(n_prefer, len(pool)), rng
        )
        prefer = sorted((by_term[t] for t in prefer_terms), key=lambda w: w.term.lower())

        recent = sorted(seen.items(), key=lambda kv: kv[1].get("last", 0), reverse=True)
        avoid = [t for t, _ in recent[:n_avoid]]
        return prefer, sorted(avoid, key=str.lower)

    def record_kept(self, target: str, sentence: str, reused: list[str]) -> None:
        """Remember a candidate the user actually chose."""
        self.kept.append(
            {"target": target, "sentence": sentence, "reused": reused, "at": time.time()}
        )
        del self.kept[:-KEEP_LIMIT]

    def recent_kept(self, n: int = 4) -> list[dict]:
        return self.kept[-n:]

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


def _weighted_sample(items: list, weights: list[float], k: int, rng: random.Random) -> list:
    """Sample k distinct items with probability proportional to weight."""
    pool = list(zip(items, weights))
    out = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        if total <= 0:
            out.extend(i for i, _ in pool[:k - len(out)])
            break
        r = rng.uniform(0, total)
        upto = 0.0
        for idx, (item, weight) in enumerate(pool):
            upto += weight
            if upto >= r:
                out.append(item)
                pool.pop(idx)
                break
    return out
