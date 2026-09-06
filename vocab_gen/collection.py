"""Read vocabulary words out of an Anki collection.

A vocab word is any text on a card's Front that is *both* underlined and
italicized. That heuristic was validated against the real collection: it picks
up 767 unique terms and excludes the italic-only emphasis used on non-vocab
cards ("what does <i>brachii</i> mean?").

Everything here is read-only. The live collection is never opened or modified.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

from .exclusions import load as load_exclusions
from html.parser import HTMLParser
from pathlib import Path

# Anki separates a note's fields with this, and "::" in deck names with it too.
FIELD_SEP = "\x1f"

# Tags that never wrap content, so they must not push a frame on the tag stack.
VOID_TAGS = frozenset({"br", "img", "hr", "input", "meta", "link", "source"})

ITALIC_TAGS = frozenset({"i", "em"})
UNDERLINE_TAGS = frozenset({"u", "ins"})

DEFAULT_PROFILE = Path.home() / "Library/Application Support/Anki2/User 1"


class _StyledRunParser(HTMLParser):
    """Collect text that is simultaneously underlined and italicized.

    Terms are emitted per *contiguous styled run* rather than per note. That
    distinction matters: a single note may underline two unrelated words
    ("hilt" and "scabbard"), which must stay separate, while a phrase like
    "sui generis" sits inside one element and must stay whole.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, bool, bool]] = []
        self._buf: list[str] = []
        self.terms: list[str] = []

    def _styled(self) -> bool:
        return any(f[1] for f in self._stack) and any(f[2] for f in self._stack)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf.clear()
        # Anki editing leaves stray quotes/punctuation inside the styled span.
        text = text.strip("\"'“”‘’ \t,.;:!?()[]")
        if text:
            self.terms.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in VOID_TAGS:
            if self._styled():
                self._buf.append(" ")
            return
        was_styled = self._styled()
        style = (dict(attrs).get("style") or "").lower()
        self._stack.append(
            (
                tag,
                tag in UNDERLINE_TAGS or "underline" in style,
                tag in ITALIC_TAGS or "italic" in style,
            )
        )
        # Entering the styled region ends whatever preceded it.
        if was_styled and not self._styled():
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        was_styled = self._styled()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break
        if was_styled and not self._styled():
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._styled():
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def terms_in_html(html: str) -> list[str]:
    """Return every underlined+italicized run in one Front field."""
    parser = _StyledRunParser()
    parser.feed(html.replace("&nbsp;", " ").replace("\xa0", " "))
    parser.close()
    return parser.terms


# When a card carries no FSRS state — new, or scheduled before FSRS was enabled
# — its forgetting probability is estimated instead. Anchored near the observed
# mean so an unknown card is neither favoured nor buried.
UNKNOWN_FORGETTING = 0.05
FORGETTING_FLOOR = 0.002  # never exactly zero, or a word could never be drawn


@dataclass(frozen=True)
class VocabWord:
    """A deck word plus what Anki knows about how well it is actually known."""

    term: str
    gloss: str = ""
    lapses: int = 0
    ivl: int = 0  # current interval in days; 0 means new or relearning
    reps: int = 0
    # FSRS memory state, straight from Anki. Absent on cards it has not scheduled.
    stability: float = 0.0  # days until recall falls to 90%
    difficulty: float = 0.0  # 1-10; how hard this card is for this learner
    decay: float = 0.0  # per-card shape of the forgetting curve
    last_review: float = 0.0  # unix seconds
    # When the note was made. Anki's note id *is* its creation time in ms, and
    # it is the only record of when a word entered the deck — which is how the
    # learner's own difficulty frontier can be seen to move.
    created: float = 0.0  # unix seconds

    @property
    def has_memory_state(self) -> bool:
        return self.stability > 0 and self.decay > 0 and self.last_review > 0

    def retrievability(self, now: float | None = None) -> float | None:
        """Probability of recalling this word right now, per FSRS.

        Stability is defined as the interval at which recall falls to 90%, so
        the factor is whatever makes R(S) equal 0.9 for this card's decay:

            R(t) = (1 + F * t/S) ** -decay,  F = 0.9 ** (-1/decay) - 1

        Returns None when Anki has not scheduled the card, rather than guessing.
        """
        if not self.has_memory_state:
            return None
        elapsed = max(0.0, ((now or time.time()) - self.last_review) / 86400.0)
        factor = 0.9 ** (-1.0 / self.decay) - 1.0
        return (1.0 + factor * elapsed / self.stability) ** (-self.decay)

    @property
    def legacy_shakiness(self) -> float:
        """The hand-rolled score FSRS replaced, kept so the swap can be measured.

        Lapses weighted heavily, a bonus for short intervals, never-studied
        cards scored low. Plausible, but invented here rather than fitted to
        anything. Rescaled to 0-1 so it can be swapped in as a sampling weight.
        """
        score = 1.0 + 2.0 * self.lapses
        if 0 < self.ivl <= 21:
            score += 1.5
        elif 21 < self.ivl <= 60:
            score += 0.5
        return score / 12.5

    @property
    def shakiness(self) -> float:
        """Probability this word has been forgotten. Higher needs reinforcement.

        Taken from FSRS where Anki has it — a memory model fitted to this
        learner's own review history beats any hand-rolled proxy. Where it does
        not, lapses and interval stand in, on the same 0-1 scale so the two
        kinds of word can be weighed against each other.
        """
        r = self.retrievability()
        if r is not None:
            return max(1.0 - r, FORGETTING_FLOOR)
        estimate = UNKNOWN_FORGETTING + 0.03 * self.lapses
        if 0 < self.ivl <= 21:
            estimate += 0.03
        return max(min(estimate, 0.5), FORGETTING_FLOOR)


_TAGS = re.compile(r"<[^>]+>")


def _memory_state(data: str | None) -> tuple[float, float, float, float]:
    """(stability, difficulty, decay, last_review) from Anki's per-card blob."""
    try:
        d = json.loads(data or "{}")
        return (
            float(d.get("s") or 0.0),
            float(d.get("d") or 0.0),
            float(d.get("decay") or 0.0),
            float(d.get("lrt") or 0.0),
        )
    except (ValueError, TypeError):
        return (0.0, 0.0, 0.0, 0.0)


def html_to_text(html: str) -> str:
    text = _TAGS.sub(" ", html.replace("&nbsp;", " ").replace("\xa0", " "))
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()


def _unicase(a: str, b: str) -> int:
    # Anki declares several text columns COLLATE unicase. Without a stub, any
    # query touching them fails with "no query solution".
    a, b = a.lower(), b.lower()
    return (a > b) - (a < b)


@contextmanager
def snapshot(profile: Path | None = None):
    """Open a throwaway copy of the collection.

    The -wal file carries edits not yet folded into the main database, so
    copying it alongside is what makes newly added cards visible. Copying only
    collection.anki2 silently yields a stale snapshot.
    """
    profile = Path(profile or DEFAULT_PROFILE)
    src = profile / "collection.anki2"
    if not src.exists():
        raise FileNotFoundError(
            f"No Anki collection at {src}.\n"
            "Pass --profile if your Anki profile lives elsewhere."
        )
    with tempfile.TemporaryDirectory(prefix="vocab-gen-") as tmp:
        dst = Path(tmp) / "collection.anki2"
        for suffix in ("", "-wal", "-shm"):
            try:
                shutil.copy2(str(src) + suffix, str(dst) + suffix)
            except FileNotFoundError:
                pass  # -wal/-shm only exist while Anki holds the collection
        conn = sqlite3.connect(dst)
        conn.create_collation("unicase", _unicase)
        try:
            yield conn
        finally:
            conn.close()


def extract_vocab(
    deck: str = "General",
    profile: Path | None = None,
    exclude_offensive: bool = True,
) -> list[VocabWord]:
    """Every unique vocab word in `deck`, with its gloss and review state.

    Suspended cards (queue -1) are excluded. Slurs and words routinely mistaken
    for them are held back too — see `exclusions.py` for why, and pass
    exclude_offensive=False to see the unfiltered list.

    Measured at ~30 ms end to end, so callers should just call this every time
    rather than caching it.
    """
    with snapshot(profile) as conn:
        rows = conn.execute(
            """
            select distinct n.id, n.flds, c.lapses, c.ivl, c.reps, c.data
              from notes n
              join cards c on c.nid = n.id
              join decks d on d.id = c.did
             where d.name like ? and c.queue != -1
            """,
            (deck.replace("::", FIELD_SEP) + "%",),
        ).fetchall()

    seen: dict[str, VocabWord] = {}
    for _nid, flds, lapses, ivl, reps, data in rows:
        created = (_nid or 0) / 1000.0
        stability, difficulty, decay, last_review = _memory_state(data)
        fields = flds.split(FIELD_SEP)
        gloss = html_to_text(fields[1]) if len(fields) > 1 else ""
        for term in terms_in_html(fields[0]):
            key = term.lower()
            prior = seen.get(key)
            if prior is None:
                seen[key] = VocabWord(
                    term, gloss, lapses or 0, ivl or 0, reps or 0,
                    stability, difficulty, decay, last_review, created,
                )
            else:
                # The same word can sit on several notes; keep the shakiest
                # reading of it, since that is the one worth reinforcing.
                # The same word can sit on several notes; keep the shakiest
                # reading of it, since that is the one worth reinforcing.
                keep_new = prior.shakiness < VocabWord(
                    term, gloss, lapses or 0, ivl or 0, reps or 0,
                    stability, difficulty, decay, last_review, created,
                ).shakiness
                seen[key] = VocabWord(
                    prior.term,
                    prior.gloss or gloss,
                    max(prior.lapses, lapses or 0),
                    min(prior.ivl, ivl or 0) if prior.ivl and ivl else (prior.ivl or ivl or 0),
                    max(prior.reps, reps or 0),
                    stability if keep_new else prior.stability,
                    difficulty if keep_new else prior.difficulty,
                    decay if keep_new else prior.decay,
                    last_review if keep_new else prior.last_review,
                    min(prior.created, created) if prior.created else created,
                )
    words = sorted(seen.values(), key=lambda w: w.term.lower())
    if exclude_offensive:
        excluded = load_exclusions()
        words = [w for w in words if w.term.strip().lower() not in excluded]
    return words


def held_back(deck: str = "General", profile: Path | None = None) -> list[str]:
    """Deck words the prompt never sees, so the filtering is inspectable."""
    excluded = load_exclusions()
    return [
        w.term
        for w in extract_vocab(deck, profile, exclude_offensive=False)
        if w.term.strip().lower() in excluded
    ]


def extract_terms(
    deck: str = "General",
    profile: Path | None = None,
    exclude_offensive: bool = True,
) -> list[str]:
    """Just the words — the shape most callers and the prompt still want."""
    return [w.term for w in extract_vocab(deck, profile, exclude_offensive)]
