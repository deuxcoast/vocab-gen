"""Turn a generated candidate into the exact HTML the user's cards use."""

from __future__ import annotations

import html
import re

from wordfreq import zipf_frequency

from .morphology import (
    Match,
    build_index,
    content_words,
    find_all,
    match_span,
    phrase_keys,
    pos_of,
)


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


# Zipf scale: 7 is "the", 6 is "need"/"against", 3 is "bearers"/"treachery".
# Measured on the words this check actually flagged: coincidental overlaps sat at
# 5.7-6.4 and genuine giveaways at 2.6-3.8, so the boundary goes between them.
INFORMATIVE_ZIPF = 4.5


def _informative(word: str) -> bool:
    """Rare enough that sharing it with the definition is unlikely to be chance."""
    return zipf_frequency(word, "en") < INFORMATIVE_ZIPF


def gives_away_answer(sentence: str, definition: list[str], target: str) -> list[str]:
    """Words shared with the definition that actually leak the meaning.

    Sharing a content word is not enough on its own: "need" or "cultural" turn up
    in a definition and a sentence by coincidence, and flagging those buries the
    real cases and adds noise to a metric used to compare prompts.

    A shared word counts when it is rare enough to be informative, or when
    several are shared at once — one common word is a coincidence, three is the
    definition being paraphrased.
    """
    target_keys = set().union(*phrase_keys(target)) if phrase_keys(target) else set()
    defined = content_words(" ".join(definition))
    if not defined:
        return []

    shared: list[tuple[str, str]] = []
    for lemma_, surface in content_words(sentence).items():
        if lemma_ in target_keys or surface.lower() in target_keys:
            continue
        if lemma_ in defined:
            shared.append((lemma_, surface))

    informative = [surface for lemma_, surface in shared if _informative(lemma_)]
    if informative:
        return informative
    # No single word is rare, but if the sentence reproduces enough of the
    # definition it is a paraphrase regardless of how common the parts are.
    # Counting shared words is the wrong test — what matters is how much of the
    # definition came through, so a long sentence cannot accumulate coincidences.
    if shared and len(shared) / len(defined) >= PARAPHRASE_SHARE:
        return [surface for _lemma, surface in shared]
    return []


# Half the definition's content words turning up is no longer a coincidence.
PARAPHRASE_SHARE = 0.5


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


def detect_reuse(sentence: str, deck, target: str, index: dict | None = None) -> list[Match]:
    """Deck words genuinely present in the sentence, target excluded.

    Reuse used to be read off the model's own `reused` list, which measured its
    self-report rather than the sentence: a sentence reusing "marginalia" was
    credited with nothing because the model had not mentioned it. Scanning the
    deck finds what is actually there.
    """
    target_keys = set().union(*phrase_keys(target)) if phrase_keys(target) else set()
    out = []
    for m in find_all(sentence, deck, index):
        if m.term.lower() in target_keys or m.text.lower() in target_keys:
            continue
        out.append(m)
    return out


def render_front(sentence: str, surface: str, reused: list[Match]) -> str:
    """The card front: target underlined and italic, reused words italic.

    Marks up by character span rather than by search-and-replace, so a word that
    occurs twice is only styled where it was actually matched.
    """
    spans: list[tuple[int, int, str, str]] = []
    target = _target_span(sentence, surface)
    if target is not None:
        spans.append((target[0], target[1], "<i><u>", "</u></i>"))
    for m in reused:
        if target is not None and not (m.end <= target[0] or m.start >= target[1]):
            continue  # never nest inside the target
        spans.append((m.start, m.end, "<i>", "</i>"))

    out, cursor = [], 0
    for start, end, open_tag, close_tag in sorted(spans):
        if start < cursor:
            continue
        out.append(html.escape(sentence[cursor:start], quote=False))
        out.append(open_tag + html.escape(sentence[start:end], quote=False) + close_tag)
        cursor = end
    out.append(html.escape(sentence[cursor:], quote=False))
    return "".join(out)


def _target_span(sentence: str, surface: str) -> tuple[int, int] | None:
    if not surface:
        return None
    for m in find_all(sentence, [surface]):
        return (m.start, m.end)
    return None


def prepare(result, word: str, deck: list[str]) -> list[dict]:
    """Verify each candidate, and check the most basic requirement of all.

    `missing_target` catches a sentence that never uses the word it is supposed
    to teach — some models substitute synonyms instead. Such a card would carry
    no underlined word at all, so it is a hard failure rather than a warning.
    """
    out = []
    index = build_index(deck)
    for c in result.candidates:
        sentence = strip_markdown(c.sentence)
        surface = strip_markdown(c.surface_form)
        present = match_span(sentence, word) or match_span(sentence, surface)
        matches = detect_reuse(sentence, deck, word, index)
        out.append(
            {
                "sentence": sentence,
                "front_html": render_front(sentence, surface or word, matches),
                "reused": [m.term for m in matches],
                "claimed": list(c.reused),
                "unclaimed": [
                    m.term for m in matches
                    if not any(r.lower() == m.term.lower() for r in c.reused)
                ],
                "giveaway": gives_away_answer(sentence, result.definition, word),
                "missing_target": present is None,
                "wrong_sense": wrong_sense(
                    sentence, word, getattr(result, "part_of_speech", "")
                ),
            }
        )
    return out


def rank_candidates(prepared: list[dict], keep: int) -> list[dict]:
    """Order over-generated candidates and keep the best `keep`.

    Ranking rather than filtering: a hard filter on reuse can leave fewer
    candidates than asked for — at a 49% reuse rate, six generations yield three
    reusing ones only about 60% of the time — and throwing the rest away is
    worse than showing them last.

    Order is by the free programmatic signals, strongest first: a candidate that
    is unusable at all, then one that reuses nothing, then one that leaks its
    definition or uses the wrong sense. Within a tier the model's own order is
    kept, since nothing here can rank prose.
    """

    def key(index_and_candidate):
        i, c = index_and_candidate
        return (
            bool(c.get("missing_target")),  # no target word at all: last
            not bool(c.get("reused")),      # reuses something: first
            bool(c.get("wrong_sense")),
            bool(c.get("giveaway")),
            i,                              # otherwise the model's own order
        )

    ordered = [c for _i, c in sorted(enumerate(prepared), key=key)]
    return ordered[:keep] if keep > 0 else ordered
