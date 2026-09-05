"""Word-form matching, via real lemmatization.

This replaces a hand-rolled suffix stripper. That stripper worked for common
inflection but could not tell derivation from inflection except by a denylist,
had no idea what part of speech a word was, and needed a stopword list written
by hand.

Two things follow from having POS available that were impossible before: the
giveaway check can compare content words only, using a real stopword list; and
a card can be checked for *sense* — if the deck teaches the verb sense and the
sentence uses the noun, the card does not reinforce what was learned.

Matching compares a set of keys per token — the surface form and the lemma —
rather than lemmas alone. Lemmatization is context-sensitive, and a deck entry
is lemmatized with almost no context ("running" alone tags as a noun), so
insisting on lemma equality would miss real matches. Any key in common at every
position counts.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

MODEL = "en_core_web_sm"

# Content words are the ones that can leak a definition; the rest are glue.
CONTENT_POS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    pos: str
    is_stop: bool
    start: int = 0  # character offsets, so matches can be marked up in place
    end: int = 0

    @property
    def keys(self) -> frozenset[str]:
        return frozenset({self.text.lower(), self.lemma.lower()})


@functools.lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
    except ImportError:
        # Reached from grading, not from the provider call, so the setup
        # classification around API errors never sees it.
        raise SystemExit(
            "spaCy is not installed in the environment running this command.\n"
            "It is a pinned dependency, so this is usually a stale install:\n"
            "  uv sync\n"
            "  uv tool install --editable --force ~/deuxcoast/vocab-gen"
        ) from None

    try:
        # The parser and NER cost time and are not used; the tagger is required
        # because the lemmatizer is rule-based and needs the part of speech.
        return spacy.load(MODEL, disable=["parser", "ner", "senter"])
    except OSError:
        raise SystemExit(
            f"The spaCy model {MODEL!r} is not installed.\n"
            "It is a pinned dependency, so this usually means the environment is "
            "stale:\n"
            "  uv sync\n"
            "  uv tool install --editable --force ~/deuxcoast/vocab-gen"
        ) from None


@functools.lru_cache(maxsize=4096)
def analyze(text: str) -> tuple[Token, ...]:
    """Tokens of `text`, with lemma and part of speech. Cached: sentences repeat."""
    doc = _nlp()(text)
    return tuple(
        Token(t.text, t.lemma_, t.pos_, t.is_stop, t.idx, t.idx + len(t.text))
        for t in doc
        if not t.is_punct and not t.is_space
    )


def tokenize(text: str) -> list[str]:
    return [t.text for t in analyze(text)]


def lemma(word: str) -> str:
    """Lemma of a single word, for callers that just want a comparison key."""
    tokens = analyze(word)
    return tokens[0].lemma.lower() if tokens else word.lower()


def _keys(text: str) -> list[frozenset[str]]:
    return [t.keys for t in analyze(text)]


def match_span(sentence: str, term: str) -> str | None:
    """Return the text in `sentence` matching `term`, allowing inflection.

    Hyphens are opened up on a second pass: a deck entry like "pied-à-terre" is
    one token, but a compound modifier like "zephyr-like" should still credit
    "zephyr".
    """
    for text, phrase in ((sentence, term), (_open(sentence), _open(term))):
        found = _find(text, phrase)
        if found:
            return found
    return None


def _open(text: str) -> str:
    return text.replace("-", " ").replace("‑", " ")


def _find(sentence: str, term: str) -> str | None:
    target = _keys(term)
    if not target:
        return None
    tokens = analyze(sentence)
    if not tokens:
        return None
    span = len(target)
    for i in range(len(tokens) - span + 1):
        window = tokens[i : i + span]
        if all(w.keys & t for w, t in zip(window, target)):
            return " ".join(t.text for t in window)
    return None


def pos_of(sentence: str, term: str) -> str | None:
    """Part of speech the target carries *in this sentence*.

    A multi-word phrase reports the tag of its head-most content word.
    """
    target = _keys(term)
    if not target:
        return None
    tokens = analyze(sentence)
    span = len(target)
    for i in range(len(tokens) - span + 1):
        window = tokens[i : i + span]
        if all(w.keys & t for w, t in zip(window, target)):
            content = [t for t in window if t.pos in CONTENT_POS]
            return (content[-1] if content else window[-1]).pos
    return None


def content_words(text: str, minimum: int = 4) -> dict[str, str]:
    """{lemma: surface} for words that could carry meaning, stopwords dropped."""
    out: dict[str, str] = {}
    for token in analyze(text):
        if token.is_stop or token.pos not in CONTENT_POS or len(token.text) < minimum:
            continue
        out.setdefault(token.lemma.lower(), token.text)
    return out


# Retained so callers reading a phrase into comparison keys have one name for it.
def phrase_keys(term: str) -> tuple[frozenset[str], ...]:
    return tuple(_keys(term))


def same_term(a: str, b: str) -> bool:
    """Whether two terms are the same word, allowing inflection.

    Token-for-token on lemma keys, so "flagons" is "flagon" and "descried" is
    "descry", while "de facto" cannot collapse onto a single-token term. Uses
    the same keys as reuse detection, so a word excluded here is exactly a word
    that would have been excluded there — and `analyze` is cached, so keying the
    whole deck costs nothing a generation was not already paying.
    """
    ka, kb = phrase_keys(a), phrase_keys(b)
    if not ka or len(ka) != len(kb):
        return False
    return all(x & y for x, y in zip(ka, kb))


_LEGACY_TOKEN = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


@dataclass(frozen=True)
class Match:
    term: str  # the deck's own spelling
    text: str  # what actually appears in the sentence
    start: int
    end: int


def build_index(deck) -> dict:
    """Map a first-token key to the deck terms that could start with it.

    Scanning 800 terms against every sentence one at a time is wasteful; this
    turns the scan into a lookup on each sentence token instead.
    """
    index: dict[str, list[tuple[str, tuple]]] = {}
    for term in deck:
        keys = tuple(_keys(term))
        if not keys:
            continue
        for first in keys[0]:
            index.setdefault(first, []).append((term, keys))
    return index


def find_all(sentence: str, deck, index: dict | None = None) -> list[Match]:
    """Every deck word actually present in the sentence.

    The point of scanning the whole deck rather than a list of claims: a model
    reuses words it does not report. Asking it what it used measures its
    self-report, not the sentence.
    """
    index = build_index(deck) if index is None else index
    tokens = analyze(sentence)
    found: list[Match] = []
    taken: set[int] = set()

    for i, token in enumerate(tokens):
        if i in taken:
            continue
        candidates = []
        for key in token.keys:
            candidates.extend(index.get(key, ()))
        # Longest match first, so "sluice gates" beats a bare "gates".
        for term, keys in sorted(candidates, key=lambda c: -len(c[1])):
            span = len(keys)
            if i + span > len(tokens) or any(j in taken for j in range(i, i + span)):
                continue
            window = tokens[i : i + span]
            if all(w.keys & k for w, k in zip(window, keys)):
                found.append(
                    Match(term, " ".join(w.text for w in window), window[0].start, window[-1].end)
                )
                taken.update(range(i, i + span))
                break
    return found
