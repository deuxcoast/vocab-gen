import pytest

from vocab_gen.morphology import contains_form, stem, stem_phrase


@pytest.mark.parametrize(
    "a, b",
    [
        ("supplicant", "supplicants"),
        ("chasm", "chasms"),
        ("adumbrate", "adumbrated"),
        ("adumbrated", "adumbrating"),
        ("emulsify", "emulsifies"),
        ("parsimony", "parsimonies"),
        ("taciturn", "taciturnly"),
        ("run", "running"),
        ("wizened", "wizening"),
    ],
)
def test_inflections_share_a_stem(a, b):
    assert stem(a) == stem(b)


@pytest.mark.parametrize(
    "a, b",
    [
        # Derivation must NOT collapse — false reuse credit is worse than a miss.
        ("relate", "relative"),
        ("strait", "straitjacket"),
        ("pomp", "pompous"),
        ("grove", "grovel"),
        ("nexus", "next"),
    ],
)
def test_unrelated_words_keep_distinct_stems(a, b):
    assert stem(a) != stem(b)


def test_short_words_are_left_alone():
    assert stem("is") == "be"
    assert stem("ford") == "ford"


def test_double_s_is_not_a_plural():
    assert stem("class") == "class"


def test_phrase_stems_token_by_token():
    assert stem_phrase("sluice gates") == stem_phrase("sluice gate")


def test_contains_form_finds_exact():
    assert contains_form("A vast chasm opened.", "chasm") == "chasm"


def test_contains_form_finds_inflection():
    assert contains_form("Vast chasms opened.", "chasm") == "chasms"
    assert contains_form("He was adumbrating the plan.", "adumbrated") == "adumbrating"


def test_contains_form_finds_multiword_phrase():
    assert contains_form("They opened the sluice gates.", "sluice gate") == "sluice gates"


def test_contains_form_rejects_substring_of_another_word():
    assert contains_form("He wore a straitjacket.", "strait") is None


def test_contains_form_returns_none_when_absent():
    assert contains_form("Nothing relevant here.", "zephyr") is None


def test_contains_form_handles_accents_and_hyphens():
    assert contains_form("A true cinéma vérité feel.", "cinéma vérité") == "cinéma vérité"
    assert contains_form("His pied-à-terre in Lyon.", "pied-à-terre") == "pied-à-terre"
