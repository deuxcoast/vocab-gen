"""Turn a generated candidate into the exact HTML the user's cards use."""

from __future__ import annotations

import html
import re

from .morphology import contains_form, stem, tokenize


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
    canonical: dict[tuple[str, ...], str] = {}
    for term in deck:
        canonical.setdefault(tuple(stem(t) for t in tokenize(term)), term)

    out: list[str] = []
    for word in claimed:
        key = tuple(stem(t) for t in tokenize(word))
        term = canonical.get(key)
        if term is None or term in out:
            continue
        if contains_form(sentence, term) or contains_form(sentence, word):
            out.append(term)
    return out


def unknown_claims(claimed: list[str], deck) -> list[str]:
    """Claimed reuses that match no deck word at all — i.e. invented.

    Must use the same inflection-aware lookup as verified_reuse: judging
    membership literally would count "supplicant" against a deck holding
    "supplicants" as a hallucination, inflating the rate for any model that
    inflects its reuses.
    """
    known = {tuple(stem(t) for t in tokenize(term)) for term in deck}
    return [w for w in claimed if tuple(stem(t) for t in tokenize(w)) not in known]


# Words too common to signal that a definition has leaked into the sentence.
_STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being
that this these those it its his her their our your my you he she they we i not no nor
who whom which what when where why how all any both each few more most other some such
only own same so than too very can will just don should now one two something someone
person people thing things way ways used using use often usually typically especially
""".split())


def gives_away_answer(sentence: str, definition: list[str], target: str) -> list[str]:
    """Content words shared by the sentence and the word's own definition.

    A card stops testing recall if the sentence hands over the meaning — the
    prompt forbids glossing the target, but nothing checked until now. Returns
    the offending words so a candidate can be flagged rather than silently
    dropped; short overlaps are often innocent.
    """
    target_stems = {stem(t) for t in tokenize(target)}
    def_stems: dict[str, str] = {}
    for line in definition:
        for token in tokenize(line):
            low = token.lower()
            if low in _STOPWORDS or len(low) < 4:
                continue
            def_stems.setdefault(stem(token), token)

    hits: list[str] = []
    for token in tokenize(sentence):
        st = stem(token)
        if st in target_stems or st not in def_stems:
            continue
        if token not in hits:
            hits.append(token)
    return hits


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
        present = contains_form(sentence, word) or contains_form(sentence, surface)
        out.append(
            {
                "sentence": sentence,
                "front_html": wrap_target(sentence, surface),
                "reused": verified_reuse(sentence, c.reused, deck),
                "giveaway": gives_away_answer(sentence, result.definition, word),
                "missing_target": present is None,
            }
        )
    return out
