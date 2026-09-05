"""Retrievability from Anki's own FSRS memory state."""

import time

import pytest

from vocab_gen.collection import FORGETTING_FLOOR, VocabWord, _memory_state

DAY = 86400.0


def card(stability=10.0, decay=0.135, days_ago=0.0, **kw):
    return VocabWord(
        "w", stability=stability, decay=decay,
        last_review=time.time() - days_ago * DAY, **kw
    )


def test_recall_is_ninety_percent_at_one_stability():
    """The definition of stability, and so the check that the curve is right."""
    for decay in (0.1, 0.135, 0.2, 0.5):
        for stability in (1.0, 10.0, 450.0):
            r = card(stability=stability, decay=decay, days_ago=stability).retrievability()
            assert r == pytest.approx(0.9, abs=1e-6), (decay, stability)


def test_recall_is_certain_immediately_after_review():
    assert card(days_ago=0).retrievability() == pytest.approx(1.0)


def test_recall_falls_as_time_passes():
    fresh = card(days_ago=1).retrievability()
    stale = card(days_ago=30).retrievability()
    assert 0 < stale < fresh <= 1.0


def test_a_more_stable_card_is_better_recalled_at_the_same_age():
    weak = card(stability=5.0, days_ago=20).retrievability()
    strong = card(stability=400.0, days_ago=20).retrievability()
    assert strong > weak


def test_recall_stays_within_zero_and_one():
    for days in (0, 1, 100, 10_000):
        r = card(stability=2.0, days_ago=days).retrievability()
        assert 0.0 < r <= 1.0, days


def test_no_memory_state_returns_none_rather_than_guessing():
    assert VocabWord("w").retrievability() is None
    assert VocabWord("w", stability=10).retrievability() is None  # decay missing too
    assert VocabWord("w").has_memory_state is False


# --- how it feeds word selection --------------------------------------------


def test_shakiness_is_the_probability_of_having_forgotten():
    c = card(stability=10.0, days_ago=10)
    assert c.shakiness == pytest.approx(1 - c.retrievability())
    assert c.shakiness == pytest.approx(0.1, abs=1e-6)  # R was 0.9


def test_a_well_known_word_still_has_a_nonzero_chance_of_selection():
    """A floor, or a word with R at 1.0 could never be drawn at all."""
    assert card(stability=400.0, days_ago=0).shakiness == FORGETTING_FLOOR


def test_words_without_memory_state_fall_back_to_lapses_and_interval():
    lapsed = VocabWord("a", lapses=3)
    solid = VocabWord("b", ivl=365)
    assert lapsed.shakiness > solid.shakiness
    assert 0 < solid.shakiness < 1 and 0 < lapsed.shakiness < 1


def test_the_fallback_is_on_the_same_scale_as_the_real_thing():
    """Both feed one weighted sample, so they must be comparable."""
    estimated = VocabWord("a", lapses=2).shakiness
    measured = card(stability=10.0, days_ago=10).shakiness
    assert 0.0 <= estimated <= 0.5 and 0.0 <= measured <= 1.0


def test_fsrs_beats_the_fallback_when_both_are_available():
    """A card with lapses but freshly reviewed is not shaky, whatever the fallback says."""
    fresh_but_lapsed = card(stability=200.0, days_ago=0, lapses=5)
    assert fresh_but_lapsed.shakiness == FORGETTING_FLOOR


# --- parsing Anki's blob ----------------------------------------------------


def test_memory_state_is_read_from_the_card_blob():
    blob = '{"pos":1,"s":6.6062,"d":8.078,"dr":0.9,"decay":0.135,"lrt":1788453751}'
    assert _memory_state(blob) == (6.6062, 8.078, 0.135, 1788453751.0)


@pytest.mark.parametrize("blob", [None, "", "{}", "not json", '{"s":"bad"}'])
def test_a_missing_or_broken_blob_is_not_an_error(blob):
    assert _memory_state(blob) == (0.0, 0.0, 0.0, 0.0)
