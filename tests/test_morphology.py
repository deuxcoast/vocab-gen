"""Lemmatization-backed matching.

These are the same behaviours the hand-rolled stemmer was tested for, so the
instrument change is checked against the old contract rather than a new one.
"""

import pytest

from vocab_gen.morphology import (
    analyze,
    content_words,
    lemma,
    match_span,
    phrase_keys,
    pos_of,
    tokenize,
)


@pytest.mark.parametrize(
    "sentence, term, expected",
    [
        ("Vast chasms opened.", "chasm", "chasms"),
        ("A lone supplicant waited.", "supplicants", "supplicant"),
        ("He was adumbrating the plan.", "adumbrated", "adumbrating"),
        ("She emulsifies the dressing.", "emulsify", "emulsifies"),
        ("His taciturn manner.", "taciturn", "taciturn"),
        ("They opened the sluice gate.", "sluice gates", "sluice gate"),
        ("A true cinéma vérité feel.", "cinéma vérité", "cinéma vérité"),
    ],
)
def test_matches_across_inflection(sentence, term, expected):
    assert match_span(sentence, term) == expected


@pytest.mark.parametrize(
    "sentence, term",
    [
        # Derivation must not collapse: false reuse credit is worse than a miss.
        ("He wore a straitjacket.", "strait"),
        ("Her pompous manner.", "pomp"),
        ("He had to grovel.", "grove"),
        ("Nothing relevant here.", "zephyr"),
    ],
)
def test_does_not_match_merely_similar_words(sentence, term):
    assert match_span(sentence, term) is None


def test_hyphenated_compound_still_credits_the_base_word():
    assert match_span("A zephyr-like breeze.", "zephyr") == "zephyr"


def test_hyphenated_deck_entry_is_found():
    assert match_span("His pied-à-terre in Lyon.", "pied-à-terre") is not None


def test_lemma_of_a_single_word():
    assert lemma("chasms") == "chasm"
    assert lemma("supplicants") == "supplicant"


def test_tokenize_drops_punctuation():
    assert tokenize("Vast chasms, opened!") == ["Vast", "chasms", "opened"]


def test_analyze_is_cached_so_repeated_sentences_are_cheap():
    a = analyze("The governor remained obdurate.")
    b = analyze("The governor remained obdurate.")
    assert a is b


def test_phrase_keys_carry_both_surface_and_lemma():
    keys = phrase_keys("chasms")
    assert "chasms" in keys[0] and "chasm" in keys[0]


# --- part of speech, which the stemmer could not do -------------------------


def test_pos_distinguishes_senses_of_the_same_word():
    assert pos_of("His countenance darkened.", "countenance") == "NOUN"
    assert pos_of("She would not countenance it.", "countenance") == "VERB"


def test_pos_returns_none_when_absent():
    assert pos_of("Nothing here.", "countenance") is None


def test_content_words_drop_stopwords_and_glue():
    out = content_words("A process that can't be stopped.")
    assert "process" in out and "stop" in out
    assert "be" not in out and "that" not in out


def test_content_words_are_keyed_by_lemma():
    assert "refuse" in content_words("refusing to change")


def test_a_missing_spacy_install_explains_itself(monkeypatch):
    """This path is reached during grading, so the API-error handling misses it."""
    import builtins

    from vocab_gen import morphology

    morphology._nlp.cache_clear()
    real_import = builtins.__import__

    def no_spacy(name, *a, **kw):
        if name == "spacy":
            raise ImportError("No module named 'spacy'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_spacy)
    with pytest.raises(SystemExit) as exc:
        morphology._nlp()
    assert "uv tool install" in str(exc.value)
    morphology._nlp.cache_clear()
