"""Crude English inflection handling.

Deliberately not a real stemmer library: this only needs to decide whether two
surface forms are the same word ("supplicant"/"supplicants",
"adumbrate"/"adumbrated"), which inflectional suffix-stripping handles. Rules
stop at inflection and never touch derivation — collapsing "relate"/"relative"
would produce false reuse credit, which is worse than missing a match.
"""

from __future__ import annotations

import re

VOWELS = set("aeiou")

# Words whose stem must not be truncated by the generic rules below.
_IRREGULAR = {
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "has": "have", "had": "have", "having": "have",
    "men": "man", "women": "woman", "children": "child", "teeth": "tooth",
    "feet": "foot", "geese": "goose", "mice": "mouse", "people": "person",
}


def stem(word: str) -> str:
    """Reduce one token to a comparison key."""
    w = word.lower().strip("'’")
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) <= 3:
        return w

    for suffix, minimum in (("ies", 4), ("ied", 4), ("ier", 4), ("iest", 5)):
        if w.endswith(suffix) and len(w) > minimum:
            return w[: -len(suffix)] + "y"

    for suffix in ("ing", "edly", "ed", "es", "s", "ly"):
        if not w.endswith(suffix):
            continue
        base = w[: -len(suffix)]
        if len(base) < 3:
            continue
        if suffix == "s" and w.endswith("ss"):
            continue  # "class" is not a plural
        if suffix == "es" and not base.endswith(("s", "x", "z", "ch", "sh")):
            # "boxes" -> "box", but "gates" -> "gate", not "gat"
            base = w[:-1]
        # "running" -> "runn" -> "run"
        if (
            suffix in ("ing", "ed")
            and len(base) > 3
            and base[-1] == base[-2]
            and base[-1] not in VOWELS
        ):
            base = base[:-1]
        return _drop_final_e(base)
    return _drop_final_e(w)


def _drop_final_e(w: str) -> str:
    """Collapse the silent-e alternation: adumbrate/adumbrated -> adumbrat.

    Applied to every stem so both sides of a comparison land in the same place.
    """
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


_TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def stem_phrase(phrase: str) -> tuple[str, ...]:
    """A multi-word deck entry compares token-by-token ('sluice gates')."""
    return tuple(stem(t) for t in tokenize(phrase))


def contains_form(sentence: str, term: str) -> str | None:
    """Return the surface text in `sentence` matching `term`, allowing inflection.

    Returns None when the term does not appear in any inflected form.
    """
    # Hyphens are kept inside tokens so "pied-à-terre" stays one word, but that
    # also welds compound modifiers together ("zephyr-like"). Try the sentence
    # as written first, then again with hyphens opened up.
    for text, split_hyphens in ((sentence, False), (sentence, True)):
        found = _find(text, term, split_hyphens)
        if found:
            return found
    return None


def _find(sentence: str, term: str, split_hyphens: bool) -> str | None:
    if split_hyphens:
        sentence = sentence.replace("-", " ").replace("\u2011", " ")
        term = term.replace("-", " ")
    target = stem_phrase(term)
    if not target:
        return None
    tokens = _TOKEN.findall(sentence)
    if not tokens:
        return None
    stems = [stem(t) for t in tokens]
    span = len(target)
    for i in range(len(stems) - span + 1):
        if tuple(stems[i : i + span]) == target:
            return " ".join(tokens[i : i + span])
    return None
