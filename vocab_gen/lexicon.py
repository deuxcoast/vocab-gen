"""Which words are worth learning next, and which of yours already qualify.

The goal is vocabulary the learner does not comfortably know yet but that is
common in educated speech and writing. Three things were tried for that, and
only the simplest survived measurement.

**Curated lists beat a scorer.** Sixteen independent GRE-style vocabulary lists
agree on 9,552 words; 75% of this deck is already on one. Academic lists were
tested first and are pitched far lower — the Academic Word List has *zero*
overlap with the deck and its top suggestions are `typically`, `stairway`,
`forehead`. A distributional scorer built on embeddings was also tested; ranked
inside the same band it returned `abidingness`, `comprehensibleness`,
`disadvantageousness` — derivational forms nobody writes. The lists exclude
those by construction, because they were built from words that actually occur.

**Population prevalence sets the neighbourhood, not the frontier.** Prevalence
— the share of people who know a word — tracks this learner's own lapse rate
(mean lapses 0.47 / 0.32 / 0.23 across prevalence thirds), so it is a real
signal. But it cannot resolve one person's knowledge: this learner knows
`alacrity` (+0.67) and hesitates over `imperturbable` (+0.84). Ordering by
prevalence alone proposes words they already know.

**So the frontier is asked, not inferred.** `calibration_sample` draws words
across the range, the learner marks known or unknown, and `frontier` fits the
level where they are half-and-half. Until enough marks exist the band falls back
to the deck's own recent additions, which is weaker and is reported as such.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .morphology import lemma

DATA = Path(__file__).resolve().parent / "data"
VERSION = 1
_LOCK = threading.Lock()

# Below this many marks the fitted frontier is noise, so the deck-derived band
# is used instead. Chosen so a single calibration pass clears it.
MIN_MARKS = 20
# A word on only one of sixteen lists is closer to that compiler's taste than to
# any consensus, and those entries are dominated by proper nouns and typos.
MIN_LISTS = 2


# ------------------------------------------------------------------ data ---
@lru_cache(maxsize=1)
def word_lists() -> dict[str, int]:
    """Word -> how many of the sixteen curated lists contain it."""
    with gzip.open(DATA / "wordlists.csv.gz", "rt", encoding="utf-8") as fh:
        return {r["word"]: int(r["n_lists"]) for r in csv.DictReader(fh)}


@lru_cache(maxsize=1)
def prevalence() -> dict[str, float]:
    """Word -> probit of the share of people who report knowing it."""
    with gzip.open(DATA / "prevalence.csv.gz", "rt", encoding="utf-8") as fh:
        return {r["word"]: float(r["prevalence"]) for r in csv.DictReader(fh)}


def deck_keys(vocab) -> set[str]:
    """Deck terms plus their lemmas, so `flagons` also covers `flagon`."""
    out: set[str] = set()
    for w in vocab:
        term = w.term.strip().lower()
        out.add(term)
        if " " not in term:
            out.add(lemma(term))
    return out


# ------------------------------------------------------- what you've said ---
def default_path() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "vocab-gen" / "vocabulary.json"


class Marks:
    """Words the learner has said they do or do not already know.

    Kept apart from usage history: that records what the tool did, this records
    what the learner told it. Only the second can locate a personal frontier.
    """

    def __init__(self, path: Path, known: dict | None = None, unknown: dict | None = None):
        self.path = path
        self.known: dict[str, float] = known or {}
        self.unknown: dict[str, float] = unknown or {}

    @classmethod
    def load(cls, path: Path | None = None) -> "Marks":
        path = Path(path or default_path())
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version") == VERSION:
                return cls(path, raw.get("known") or {}, raw.get("unknown") or {})
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return cls(path)

    def save(self) -> None:
        payload = {"version": VERSION, "known": self.known, "unknown": self.unknown}
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=1, sort_keys=True)
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    def mark(self, words, known: bool) -> None:
        now = time.time()
        for word in words:
            key = word.strip().lower()
            if not key:
                continue
            # A later answer wins, and a word never sits in both.
            (self.unknown if known else self.known).pop(key, None)
            (self.known if known else self.unknown)[key] = now

    def __len__(self) -> int:
        return len(self.known) + len(self.unknown)


# --------------------------------------------------------- the frontier ---
@dataclass(frozen=True)
class Band:
    floor: float
    ceiling: float
    source: str  # how it was derived, so a weak basis is never silently used
    n_marks: int = 0


def recent_band(vocab, fraction: float = 0.25) -> Band:
    """Fallback: the difficulty of what this learner has added most recently.

    Anki's note id is its creation time, so the deck records its own history.
    On the real deck the median prevalence of new additions has fallen from
    +0.79 to +0.55, which is the frontier moving. Using the whole deck instead
    would answer with where the learner was years ago.
    """
    prev = prevalence()
    dated = sorted(
        ((w.created, prev[w.term.strip().lower()]) for w in vocab
         if w.created and w.term.strip().lower() in prev),
        key=lambda t: -t[0],
    )
    if len(dated) < 20:
        return Band(-1.0, 1.0, "default (too little deck history)")
    recent = sorted(p for _c, p in dated[: max(20, int(len(dated) * fraction))])
    floor = recent[len(recent) // 10]
    ceiling = recent[len(recent) // 2]  # median: at least as hard as a typical recent add
    return Band(floor, ceiling, f"recent {int(fraction*100)}% of deck additions")


def frontier(marks: Marks, vocab=None) -> Band:
    """Where the learner is half-and-half, fitted from their own answers.

    Deliberately not a logistic regression: with a few dozen marks the fit is
    dominated by its own assumptions, and a crossover point over sorted marks is
    both robust and explainable. Words above the ceiling are ones they reliably
    know, so proposing them wastes attention.
    """
    prev = prevalence()
    pts = [(prev[w], 1) for w in marks.known if w in prev]
    pts += [(prev[w], 0) for w in marks.unknown if w in prev]
    if len(pts) < MIN_MARKS:
        band = recent_band(vocab or [])
        return Band(band.floor, band.ceiling, band.source, len(pts))

    pts.sort()
    # Sweep for the split where "known above, unknown below" is most consistent.
    best, best_at = -1.0, pts[len(pts) // 2][0]
    for i in range(2, len(pts) - 2):
        below = pts[:i]
        above = pts[i:]
        score = (sum(1 for _p, k in below if not k) / len(below)
                 + sum(1 for _p, k in above if k) / len(above))
        if score > best:
            best, best_at = score, pts[i][0]
    known_p = [p for p, k in pts if k]
    floor = min(pts)[0] if not known_p else min(min(known_p) - 1.0, min(pts)[0])
    return Band(floor, best_at, "fitted from your marks", len(pts))


# ----------------------------------------------------------- the answers ---
@dataclass(frozen=True)
class Candidate:
    word: str
    n_lists: int
    prevalence: float


def suggest(vocab, marks: Marks | None = None, n: int = 30,
            band: Band | None = None, rng: random.Random | None = None) -> list[Candidate]:
    """Words on the curated lists that this learner lacks, hardest-consensus first.

    Ordered by list agreement and then by rarity. Sampling is not used: unlike
    the preferred-word list in `history`, there is no coverage goal here — the
    learner works down a ranking and marks as they go.
    """
    lists, prev = word_lists(), prevalence()
    have = deck_keys(vocab)
    known = set(marks.known) if marks else set()
    band = band or frontier(marks or Marks(default_path()), vocab)

    out = [
        Candidate(w, n_lists, prev[w])
        for w, n_lists in lists.items()
        if n_lists >= MIN_LISTS and w not in have and w not in known
        and w in prev and band.floor <= prev[w] <= band.ceiling
    ]
    out.sort(key=lambda c: (-c.n_lists, c.prevalence))
    return out[:n]


def calibration_sample(vocab, n: int = 40, rng: random.Random | None = None,
                       span: tuple[float, float] = (-1.5, 1.5)) -> list[str]:
    """Words spread across the plausible range, for the learner to mark.

    Spread rather than ranked on purpose: a sample taken from one end locates
    nothing, because the fit needs words on both sides of the frontier.

    `span` bounds it because a question whose answer is never in doubt teaches
    the fit nothing — asking this learner about `dent` or `weary` spends their
    attention to confirm what prevalence already says. The range is wide enough
    to bracket any plausible frontier and no wider.
    """
    rng = rng or random.Random()
    lists, prev = word_lists(), prevalence()
    have = deck_keys(vocab)
    pool = [w for w, k in lists.items()
            if k >= MIN_LISTS and w not in have and w in prev
            and span[0] <= prev[w] <= span[1]]
    if not pool:
        return []
    pool.sort(key=lambda w: prev[w])
    step = len(pool) / n
    picked = []
    for i in range(n):
        lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
        picked.append(rng.choice(pool[lo:hi]))
    return picked


def deck_report(vocab, marks: Marks | None = None) -> dict:
    """Where the existing deck sits against the curated lists and prevalence."""
    lists, prev = word_lists(), prevalence()
    have = deck_keys(vocab)
    on_list = sorted(w for w in have if w in lists)
    scored = sorted(prev[w] for w in have if w in prev)
    band = frontier(marks or Marks(default_path()), vocab)
    remaining = [
        w for w, k in lists.items()
        if k >= MIN_LISTS and w not in have and w in prev
        and band.floor <= prev[w] <= band.ceiling
        and not (marks and w in marks.known)
    ]
    return {
        "deck": len(vocab),
        "on_list": len(on_list),
        "matched_prevalence": len(scored),
        "median_prevalence": scored[len(scored) // 2] if scored else None,
        "band": band,
        "remaining": len(remaining),
        "list_total": sum(1 for k in lists.values() if k >= MIN_LISTS),
    }
