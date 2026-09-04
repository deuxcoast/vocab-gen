from vocab_gen.render import back_html, wrap_target


def test_wraps_first_occurrence_in_anki_nesting():
    assert wrap_target("The nexus was clear.", "nexus") == (
        "The <i><u>nexus</u></i> was clear."
    )


def test_wraps_inflected_surface_form():
    out = wrap_target("Vast chasms opened.", "chasms")
    assert out == "Vast <i><u>chasms</u></i> opened."


def test_wraps_multiword_phrase():
    out = wrap_target("It was sui generis, really.", "sui generis")
    assert "<i><u>sui generis</u></i>" in out


def test_case_insensitive_fallback():
    out = wrap_target("Bucolic charm.", "bucolic")
    assert out == "<i><u>Bucolic</u></i> charm."


def test_only_first_occurrence_is_wrapped():
    out = wrap_target("A yoke is a yoke.", "yoke")
    assert out.count("<i><u>") == 1


def test_missing_surface_form_leaves_sentence_intact():
    out = wrap_target("No such word here.", "absent")
    assert out == "No such word here."


def test_html_in_sentence_is_escaped():
    out = wrap_target("Tom & Jerry <fought> a nexus.", "nexus")
    assert "&amp;" in out and "&lt;fought&gt;" in out
    assert "<i><u>nexus</u></i>" in out


def test_back_html_matches_card_format():
    assert back_html(["Unyielding.", "Can't be stopped."]) == (
        "<ul><li>Unyielding.</li><li>Can't be stopped.</li></ul>"
    )


from vocab_gen.render import verified_reuse

KNOWN = ["zephyr", "strait", "parsimonious", "cinéma vérité", "pied-à-terre"]


def test_credits_a_word_that_is_really_there():
    assert verified_reuse("Carried on a zephyr.", ["zephyr"], KNOWN) == ["zephyr"]


def test_drops_a_word_the_model_claimed_but_did_not_use():
    assert verified_reuse("The district remained firm.", ["parsimonious"], KNOWN) == []


def test_drops_a_word_that_is_not_in_the_deck():
    assert verified_reuse("A bogus claim.", ["bogus"], KNOWN) == []


def test_match_is_case_insensitive():
    assert verified_reuse("Strait of Hormuz.", ["strait"], KNOWN) == ["strait"]


def test_does_not_match_inside_a_longer_word():
    # 'strait' must not be credited by 'straitjacket'
    assert verified_reuse("He wore a straitjacket.", ["strait"], KNOWN) == []


def test_accented_and_hyphenated_terms_match():
    assert verified_reuse("A true cinéma vérité feel.", ["cinéma vérité"], KNOWN) == [
        "cinéma vérité"
    ]
    assert verified_reuse("His pied-à-terre in Lyon.", ["pied-à-terre"], KNOWN) == [
        "pied-à-terre"
    ]


def test_hyphen_boundary_is_respected():
    assert verified_reuse("A zephyr-like breeze.", ["zephyr"], KNOWN) == ["zephyr"]


# --- inflection-aware reuse -------------------------------------------------

DECK = ["supplicants", "chasms", "adumbrated", "strait", "sluice gates", "zephyr"]


def test_credits_an_inflected_form_of_a_deck_word():
    """Deck has 'supplicants'; the sentence writes 'supplicant'."""
    assert verified_reuse("A lone supplicant waited.", ["supplicant"], DECK) == [
        "supplicants"
    ]


def test_returns_the_decks_spelling_not_the_sentences():
    """History must aggregate one word to one key, not scatter by inflection."""
    assert verified_reuse("He adumbrates the plan.", ["adumbrates"], DECK) == [
        "adumbrated"
    ]


def test_multiword_phrase_inflection():
    assert verified_reuse("They opened the sluice gate.", ["sluice gate"], DECK) == [
        "sluice gates"
    ]


def test_still_rejects_a_word_not_in_the_sentence():
    assert verified_reuse("Nothing here.", ["zephyr"], DECK) == []


def test_still_rejects_a_word_not_in_the_deck():
    assert verified_reuse("A bogus mawkishness.", ["mawkishness"], DECK) == []


def test_does_not_credit_a_substring_word():
    assert verified_reuse("He wore a straitjacket.", ["strait"], DECK) == []


def test_deduplicates_repeated_claims():
    assert verified_reuse("Chasms and chasms.", ["chasms", "chasm"], DECK) == ["chasms"]


# --- answer giveaway --------------------------------------------------------

from vocab_gen.render import gives_away_answer


def test_flags_a_definition_word_appearing_in_the_sentence():
    hits = gives_away_answer(
        "The union chief was unyielding, and remained obdurate for weeks.",
        ["Unyielding; refusing to change."],
        "obdurate",
    )
    assert "unyielding" in [h.lower() for h in hits]


def test_flags_an_inflected_definition_word():
    hits = gives_away_answer(
        "His yielding manner surprised them; he was not obdurate.",
        ["Unyielding."],
        "obdurate",
    )
    assert hits == [] or "yielding" not in [h.lower() for h in hits]


def test_clean_sentence_has_no_giveaway():
    assert (
        gives_away_answer(
            "The board offered concessions, but the chief remained obdurate.",
            ["Unyielding; refusing to change."],
            "obdurate",
        )
        == []
    )


def test_target_word_itself_is_not_a_giveaway():
    assert (
        gives_away_answer(
            "He remained obdurate.", ["Obdurate means unyielding."], "obdurate"
        )
        == []
    )


def test_stopwords_and_short_words_are_ignored():
    assert (
        gives_away_answer(
            "The person used it in a way that was very odd.",
            ["A person who uses things in some way."],
            "quixotic",
        )
        == []
    )


from vocab_gen.render import unknown_claims


def test_unknown_claims_flags_only_real_inventions():
    assert unknown_claims(["mawkishness"], DECK) == ["mawkishness"]


def test_unknown_claims_does_not_punish_inflection():
    """The bug this replaces: 'supplicant' against a deck holding 'supplicants'."""
    assert unknown_claims(["supplicant"], DECK) == []
    assert unknown_claims(["adumbrates"], DECK) == []


def test_unknown_claims_and_verified_reuse_agree():
    """A claim cannot be both credited and counted as invented."""
    sentence = "A lone supplicant waited near the strait."
    claims = ["supplicant", "strait", "mawkishness"]
    credited = verified_reuse(sentence, claims, DECK)
    invented = unknown_claims(claims, DECK)
    assert not (set(credited) & set(invented))
    assert len(credited) + len(invented) == len(claims)
