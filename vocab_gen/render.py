"""Turn a generated candidate into the exact HTML the user's cards use."""

from __future__ import annotations

import html
import re


def wrap_target(sentence: str, surface_form: str) -> str:
    """Underline+italicize the target inside the sentence, Anki-style.

    Matches the nesting the existing cards use: <i><u>word</u></i>. Only the
    first occurrence is wrapped. Falls back to a case-insensitive match, and if
    the model reported a surface form that isn't actually in the sentence, the
    sentence is returned escaped but unwrapped rather than mangled.
    """
    escaped = html.escape(sentence, quote=False)
    target = html.escape(surface_form, quote=False)
    if not target:
        return escaped

    idx = escaped.find(target)
    if idx == -1:
        match = re.search(re.escape(target), escaped, re.IGNORECASE)
        if not match:
            return escaped
        idx, target = match.start(), match.group(0)

    return f"{escaped[:idx]}<i><u>{target}</u></i>{escaped[idx + len(target):]}"


def back_html(bullets: list[str]) -> str:
    """The Back field: a plain <ul> of short definition bullets."""
    items = "".join(f"<li>{html.escape(b, quote=False)}</li>" for b in bullets)
    return f"<ul>{items}</ul>"


def verified_reuse(sentence: str, claimed: list[str], known: set[str]) -> list[str]:
    """Keep only reuse claims that survive checking.

    The model is asked to report which known words it used, but a claim is only
    credited if the word is genuinely in the deck *and* genuinely present in the
    sentence. Without the second check a mis-reported word would be displayed as
    a reuse that isn't there, which is exactly the signal the user is reading.
    """
    out = []
    for word in claimed:
        if word.lower() not in known:
            continue
        # \b won't anchor against accents or hyphens, so bound on non-letters.
        pattern = rf"(?<![^\W\d_]){re.escape(word)}(?![^\W\d_])"
        if re.search(pattern, sentence, re.IGNORECASE):
            out.append(word)
    return out
