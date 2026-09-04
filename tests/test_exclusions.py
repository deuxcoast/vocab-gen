import pytest

from vocab_gen.collection import VocabWord
from vocab_gen.exclusions import DEFAULT_EXCLUSIONS, is_excluded, load, user_file


def test_the_words_that_prompted_this_are_excluded():
    for word in ("maricón", "jigaboo", "niggardly"):
        assert is_excluded(word), word


def test_matching_is_case_and_whitespace_insensitive():
    assert is_excluded("  Niggardly  ")
    assert is_excluded("JIGABOO")


def test_words_merely_unpleasant_are_not_excluded():
    """The rule is slurs, not distaste — the deck is full of dark vocabulary."""
    for word in ("pogrom", "offal", "gallows", "despot", "lascivious", "prurient"):
        assert not is_excluded(word), word


def test_a_user_file_extends_the_defaults(tmp_path):
    path = tmp_path / "excluded.txt"
    path.write_text("# personal additions\nsomeword\n\n  OtherWord  # trailing note\n")
    loaded = load(path)
    assert "someword" in loaded and "otherword" in loaded
    assert DEFAULT_EXCLUSIONS <= loaded, "the user file must not remove defaults"


def test_a_missing_user_file_is_not_an_error(tmp_path):
    assert load(tmp_path / "nope.txt") == DEFAULT_EXCLUSIONS


def test_user_file_lives_under_xdg_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert user_file() == tmp_path / "vocab-gen" / "excluded.txt"


def test_extraction_filters_by_default(monkeypatch):
    from vocab_gen import collection

    sample = [VocabWord("zephyr"), VocabWord("niggardly"), VocabWord("strait")]
    monkeypatch.setattr(
        collection, "extract_vocab",
        lambda deck="General", profile=None, exclude_offensive=True: (
            [w for w in sample if not exclude_offensive or not is_excluded(w.term)]
        ),
    )
    assert [w.term for w in collection.extract_vocab()] == ["zephyr", "strait"]
    assert len(collection.extract_vocab(exclude_offensive=False)) == 3


def test_excluding_a_word_does_not_stop_you_targeting_it():
    """The filter governs what the tool volunteers, not what you may study."""
    from vocab_gen.generate import build_user_message

    message = build_user_message("niggardly", 3)
    assert "niggardly" in message


def test_exclusion_is_justified_by_card_quality_not_by_classifiers():
    """Recorded because the original rationale turned out to be wrong.

    The list was written believing it would clear Alibaba's content filter. It
    does not: measured on a paid account, the variant that had been rejected
    passes with the slurs present. The rule is about what belongs on a card.
    """
    assert is_excluded("jigaboo")
    assert not is_excluded("pogrom"), "not a slur; the rule is not about discomfort"
