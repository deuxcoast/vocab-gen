"""Turn a generated candidate into the exact HTML the user's cards use."""

from __future__ import annotations

import html
import re

from .morphology import content_words, match_span, phrase_keys, pos_of


# Asterisk emphasis may sit inside a word; underscore emphasis may not — which
# is what keeps snake_case_name intact. Markdown itself draws the same line.
_MD_STAR = re.compile(r"(\*\*|\*)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_MD_UNDERSCORE = re.compile(
    r"(?<![A-Za-z0-9])(__|_)(?=\S)(.+?)(?<=\S)\1(?![A-Za-z0-9])", re.DOTALL
)


def strip_markdown(text: str) -> str:
    """Remove markdown emphasis a model wrapped around words.

    Some providers bold the target word or the reused vocabulary inside the
    sentence itself. The prompt asks for plain text, but that cannot be relied
    on across vendors, and literal asterisks would land on the card front.
    """
    previous = None
    while previous != text:  # ***both*** needs two passes
        previous = text
        text = _MD_STAR.sub(r"\2", text)
        text = _MD_UNDERSCORE.sub(r"\2", text)
    return text


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


def verified_reuse(sentence: str, claimed: list[str], deck) -> list[str]:
    """Keep only reuse claims that survive checking, in the deck's own spelling.

    A claim is credited when the word is genuinely in the deck *and* genuinely
    present in the sentence. Matching is inflection-aware, so a deck entry of
    "supplicants" is credited when the sentence writes "supplicant" — and the
    deck's spelling is what comes back, so usage history aggregates one word to
    one key instead of scattering across its inflections.
    """
    out: list[str] = []
    for word in claimed:
        term = _canonical(word, deck)
        if term is None or term in out:
            continue
        if match_span(sentence, term) or match_span(sentence, word):
            out.append(term)
    return out


def _canonical(word: str, deck) -> str | None:
    """The deck's own spelling of a claimed word, matching across inflection."""
    keys = phrase_keys(word)
    for term in deck:
        term_keys = phrase_keys(term)
        if len(term_keys) == len(keys) and all(a & b for a, b in zip(keys, term_keys)):
            return term
    return None


def unknown_claims(claimed: list[str], deck) -> list[str]:
    """Claimed reuses that match no deck word at all — i.e. invented.

    Must use the same inflection-aware lookup as verified_reuse: judging
    membership literally would count "supplicant" against a deck holding
    "supplicants" as a hallucination, inflating the rate for any model that
    inflects its reuses.
    """
    return [w for w in claimed if _canonical(w, deck) is None]


def gives_away_answer(sentence: str, definition: list[str], target: str) -> list[str]:
    """Content words shared by the sentence and the word's own definition.

    A card stops testing recall if the sentence hands over the meaning. Only
    content words count — real part-of-speech tagging replaces a hand-written
    stopword list, so "used", "way" and "person" no longer have to be guessed at.
    Returns the offending words so a candidate can be flagged rather than
    dropped; short overlaps are often innocent.
    """
    target_keys = set().union(*phrase_keys(target)) if phrase_keys(target) else set()
    defined = content_words(" ".join(definition))
    if not defined:
        return []

    hits: list[str] = []
    for lemma_, surface in content_words(sentence).items():
        if lemma_ in target_keys or surface.lower() in target_keys:
            continue
        if lemma_ in defined and surface not in hits:
            hits.append(surface)
    return hits


# How the model's own part-of-speech label maps onto what spaCy tags.
POS_ALIASES = {
    "noun": {"NOUN", "PROPN"},
    "verb": {"VERB", "AUX"},
    "adjective": {"ADJ"},
    "adverb": {"ADV"},
}


def wrong_sense(sentence: str, target: str, claimed_pos: str) -> str | None:
    """Is the target used as a different part of speech than its definition?

    A deck entry for the verb `countenance` is not reinforced by a sentence
    using the noun. Returns the tag actually used when it conflicts, else None.
    Anything outside the four main classes is not judged.
    """
    expected = None
    label = (claimed_pos or "").strip().lower()
    for name, tags in POS_ALIASES.items():
        if label.startswith(name):
            expected = tags
            break
    if expected is None:
        return None
    actual = pos_of(sentence, target)
    if actual is None or actual in expected:
        return None
    return actual


def prepare(result, word: str, deck: list[str]) -> list[dict]:
    """Verify each candidate, and check the most basic requirement of all.

    `missing_target` catches a sentence that never uses the word it is supposed
    to teach — some models substitute synonyms instead. Such a card would carry
    no underlined word at all, so it is a hard failure rather than a warning.
    """
    out = []
    for c in result.candidates:
        sentence = strip_markdown(c.sentence)
        surface = strip_markdown(c.surface_form)
        present = match_span(sentence, word) or match_span(sentence, surface)
        out.append(
            {
                "sentence": sentence,
                "front_html": wrap_target(sentence, surface),
                "reused": verified_reuse(sentence, c.reused, deck),
                "giveaway": gives_away_answer(sentence, result.definition, word),
                "missing_target": present is None,
                "wrong_sense": wrong_sense(
                    sentence, word, getattr(result, "part_of_speech", "")
                ),
            }
        )
    return out
