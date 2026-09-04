"""Read vocabulary words out of an Anki collection.

A vocab word is any text on a card's Front that is *both* underlined and
italicized. That heuristic was validated against the real collection: it picks
up 767 unique terms and excludes the italic-only emphasis used on non-vocab
cards ("what does <i>brachii</i> mean?").

Everything here is read-only. The live collection is never opened or modified.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VocabWord:
    """A deck word plus what Anki knows about how well it is actually known."""

    term: str
    gloss: str = ""
    lapses: int = 0
    ivl: int = 0  # current interval in days; 0 means new or relearning
    reps: int = 0

    @property
    def shakiness(self) -> float:
        """How much this word still needs reinforcement. Higher is shakier.

        Lapses dominate: forgetting a card you have already learned is the
        clearest signal it has not stuck. A short interval is a weaker hint that
        it is still bedding in. Never-studied cards score low on purpose — they
        are already scheduled to come up on their own.
        """
        score = 1.0 + 2.0 * self.lapses
        if 0 < self.ivl <= 21:
            score += 1.5
        elif 21 < self.ivl <= 60:
            score += 0.5
        return score


_TAGS = re.compile(r"<[^>]+>")


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


def extract_vocab(deck: str = "General", profile: Path | None = None) -> list[VocabWord]:
    """Every unique vocab word in `deck`, with its gloss and review state.

    Suspended cards (queue -1) are excluded. Measured at ~30 ms end to end, so
    callers should just call this every time rather than caching it.
    """
    with snapshot(profile) as conn:
        rows = conn.execute(
            """
            select distinct n.id, n.flds, c.lapses, c.ivl, c.reps
              from notes n
              join cards c on c.nid = n.id
              join decks d on d.id = c.did
             where d.name like ? and c.queue != -1
            """,
            (deck.replace("::", FIELD_SEP) + "%",),
        ).fetchall()

    seen: dict[str, VocabWord] = {}
    for _nid, flds, lapses, ivl, reps in rows:
        fields = flds.split(FIELD_SEP)
        gloss = html_to_text(fields[1]) if len(fields) > 1 else ""
        for term in terms_in_html(fields[0]):
            key = term.lower()
            prior = seen.get(key)
            if prior is None:
                seen[key] = VocabWord(term, gloss, lapses or 0, ivl or 0, reps or 0)
            else:
                # The same word can sit on several notes; keep the shakiest
                # reading of it, since that is the one worth reinforcing.
                seen[key] = VocabWord(
                    prior.term,
                    prior.gloss or gloss,
                    max(prior.lapses, lapses or 0),
                    min(prior.ivl, ivl or 0) if prior.ivl and ivl else (prior.ivl or ivl or 0),
                    max(prior.reps, reps or 0),
                )
    return sorted(seen.values(), key=lambda w: w.term.lower())


def extract_terms(deck: str = "General", profile: Path | None = None) -> list[str]:
    """Just the words — the shape most callers and the prompt still want."""
    return [w.term for w in extract_vocab(deck, profile)]
