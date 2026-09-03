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


def extract_terms(deck: str = "General", profile: Path | None = None) -> list[str]:
    """Every unique vocab word in `deck`, case-insensitively deduped and sorted.

    Suspended cards (queue -1) are excluded. Measured at ~22 ms end to end, so
    callers should just call this every time rather than caching it.
    """
    with snapshot(profile) as conn:
        rows = conn.execute(
            """
            select distinct n.flds
              from notes n
              join cards c on c.nid = n.id
              join decks d on d.id = c.did
             where d.name like ? and c.queue != -1
            """,
            (deck.replace("::", FIELD_SEP) + "%",),
        ).fetchall()

    seen: dict[str, str] = {}
    for (flds,) in rows:
        front = flds.split(FIELD_SEP)[0]
        for term in terms_in_html(front):
            seen.setdefault(term.lower(), term)
    return sorted(seen.values(), key=str.lower)
