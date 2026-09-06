import pytest

from vocab_gen.render import back_html, prepare, wrap_target


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


def test_real_stopwords_are_ignored():
    """spaCy's stopword list replaces a hand-written one; glue words never count.

    The definition carries content words, so the check does run — what it must
    not do is flag the glue the two happen to share.
    """
    assert (
        gives_away_answer(
            "It was the one that they had, in a way.",
            ["The one that they had, in some impractical fashion."],
            "quixotic",
        )
        == []
    )


def test_a_definition_of_pure_glue_is_unassessable():
    """No content words means nothing to compare, however much text there is."""
    assert (
        gives_away_answer(
            "It was the one that they had, in a way.",
            ["The one that they had, in some way."],
            "quixotic",
        )
        is None
    )


def test_a_shared_content_word_is_flagged_even_if_generic():
    """The old hand-written list suppressed 'person'; it is a content word and
    a genuine overlap, so flagging it is the more honest behaviour."""
    hits = gives_away_answer(
        "The person spoke.", ["A person who does this."], "quixotic"
    )
    assert [h.lower() for h in hits] == ["person"]


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


from vocab_gen.render import strip_markdown


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("remained **obdurate**, insisting", "remained obdurate, insisting"),
        ("his *sedulous* attempts", "his sedulous attempts"),
        ("__both__ and _kinds_", "both and kinds"),
        ("***very*** emphatic", "very emphatic"),
        ("nothing to strip here", "nothing to strip here"),
    ],
)
def test_strip_markdown(raw, clean):
    assert strip_markdown(raw) == clean


def test_strip_markdown_leaves_bare_punctuation_alone():
    # A lone asterisk or an intra-word underscore is not emphasis.
    assert strip_markdown("a * b") == "a * b"
    assert strip_markdown("snake_case_name") == "snake_case_name"


class _C:
    def __init__(self, sentence, surface_form, reused=()):
        self.sentence, self.surface_form, self.reused = sentence, surface_form, list(reused)


class _R:
    definition = ["Unyielding."]

    def __init__(self, cands):
        self.candidates = cands


def test_prepare_flags_a_sentence_missing_the_target():
    """Some models substitute synonyms; that card has no word to underline."""
    r = _R([_C("The leadership remained stubborn and unyielding.", "obdurate")])
    assert prepare(r, "obdurate", DECK)[0]["missing_target"] is True


def test_prepare_accepts_an_inflected_target():
    r = _R([_C("Their obduracy was total.", "obduracy")])
    out = prepare(r, "obdurate", DECK)[0]
    assert out["missing_target"] is False


def test_prepare_accepts_the_plain_target():
    r = _R([_C("The governor remained obdurate.", "obdurate")])
    assert prepare(r, "obdurate", DECK)[0]["missing_target"] is False


def test_prepare_strips_markdown_before_checking():
    r = _R([_C("The governor remained **obdurate**.", "**obdurate**")])
    out = prepare(r, "obdurate", DECK)[0]
    assert out["missing_target"] is False
    assert "**" not in out["front_html"]


# --- sense checking, which the stemmer could not do -------------------------

from vocab_gen.render import wrong_sense


def test_flags_a_noun_used_where_the_card_teaches_a_verb():
    """A deck entry for the verb 'countenance' is not reinforced by the noun."""
    assert wrong_sense("His countenance darkened.", "countenance", "verb") == "NOUN"


def test_accepts_the_sense_the_card_teaches():
    assert wrong_sense("She would not countenance it.", "countenance", "verb") is None
    assert wrong_sense("His countenance darkened.", "countenance", "noun") is None


def test_adjective_and_adverb_are_distinguished():
    assert wrong_sense("He answered peremptorily.", "peremptorily", "adjective") == "ADV"


def test_unrecognised_part_of_speech_is_not_judged():
    assert wrong_sense("It was sui generis.", "sui generis", "phrase") is None
    assert wrong_sense("The judge was obdurate.", "obdurate", "") is None


def test_absent_target_is_not_a_sense_error():
    """Missing the word entirely is a different, harder failure."""
    assert wrong_sense("Nothing relevant.", "obdurate", "adjective") is None


def test_prepare_reports_the_sense_check():
    r = _R([_C("The governor remained obdurate.", "obdurate")])
    r.part_of_speech = "adjective"
    assert prepare(r, "obdurate", DECK)[0]["wrong_sense"] is None


# --- giveaway precision ------------------------------------------------------


def test_a_single_common_word_is_not_a_giveaway():
    """The original complaint: benign words shared by chance were being flagged."""
    for word in ("rather", "against", "over", "through", "need"):
        sentence = f"He argued {word} the proposal at length before the committee."
        definition = [f"To speak bitterly {word} something one dislikes."]
        assert gives_away_answer(sentence, definition, "inveigh") == [], word


def test_a_rare_shared_word_is_a_giveaway():
    hits = gives_away_answer(
        "He was unyielding, utterly obdurate.", ["Unyielding."], "obdurate"
    )
    assert [h.lower() for h in hits] == ["unyielding"]


def test_rarity_is_what_separates_them():
    from vocab_gen.render import INFORMATIVE_ZIPF, _informative

    assert _informative("unyielding") and _informative("bearers")
    assert not _informative("need") and not _informative("against")
    assert 3.8 < INFORMATIVE_ZIPF < 5.5  # between the two observed clusters


def test_reproducing_most_of_the_definition_is_a_giveaway():
    """Common words individually, but the definition restated in aggregate."""
    hits = gives_away_answer(
        "The new road will remove the need to detour.",
        ["To remove a need or difficulty."],
        "obviate",
    )
    assert {h.lower() for h in hits} == {"remove", "need"}


def test_one_common_word_in_a_long_definition_is_not_enough():
    """A long sentence must not accumulate coincidences into a flag."""
    assert (
        gives_away_answer(
            "The room was full of people who needed somewhere to sit and wait.",
            ["A sudden and complete change of allegiance, usually for personal gain."],
            "apostasy",
        )
        == []
    )


def test_the_target_itself_never_counts():
    assert gives_away_answer(
        "He remained obdurate.", ["Obdurate means unyielding."], "obdurate"
    ) == []


def test_an_unusable_definition_is_not_assessable_rather_than_clean():
    """This test previously asserted [] — the bug, encoded as correct behaviour.

    An empty definition means the check could not run. Recording that as "no
    giveaway found" silently counts an unmeasurable card as evidence of no leak,
    in a metric that prompts are compared on.
    """
    assert gives_away_answer("Anything at all.", [], "obdurate") is None
    assert gives_away_answer("Anything at all.", [", ", ". "], "obdurate") is None
    # A definition with real content still returns a list, empty when clean.
    assert gives_away_answer("A quiet morning.", ["Unyielding."], "obdurate") == []


# --- reuse detection, not reuse self-report ---------------------------------

from vocab_gen.render import detect_reuse, render_front

BIG_DECK = ["marginalia", "glib", "strait", "supplicants", "sluice gates", "zephyr"]


def test_finds_a_reused_word_the_model_never_mentioned():
    """The reported bug: 'marginalia' was in the deck and in the sentence, and
    was missed because the model only claimed 'glib'."""
    sentence = "She read the editor's marginalia as a glib exercise in pedantry."
    found = [m.term for m in detect_reuse(sentence, BIG_DECK, "pedantry")]
    assert set(found) == {"marginalia", "glib"}


def test_detection_ignores_what_the_model_claimed_entirely():
    sentence = "A lone supplicant crossed the strait."
    found = {m.term for m in detect_reuse(sentence, BIG_DECK, "obdurate")}
    assert found == {"supplicants", "strait"}


def test_the_target_is_never_counted_as_reuse():
    sentence = "The zephyr was a true zephyr."
    assert detect_reuse(sentence, BIG_DECK, "zephyr") == []


def test_longest_phrase_wins_over_its_parts():
    sentence = "They opened the sluice gates at dawn."
    found = [m.term for m in detect_reuse(sentence, BIG_DECK + ["gates"], "dawn")]
    assert "sluice gates" in found and "gates" not in found


def test_detection_matches_across_inflection():
    sentence = "A lone supplicant waited."
    assert [m.term for m in detect_reuse(sentence, BIG_DECK, "x")] == ["supplicants"]


# --- front rendering ---------------------------------------------------------


def test_front_italicises_reuse_and_underlines_the_target():
    sentence = "She read the marginalia as a glib exercise in pedantry."
    html_out = render_front(sentence, "pedantry", detect_reuse(sentence, BIG_DECK, "pedantry"))
    assert "<i>marginalia</i>" in html_out
    assert "<i>glib</i>" in html_out
    assert "<i><u>pedantry</u></i>" in html_out


def test_front_escapes_the_surrounding_text():
    sentence = "Tom & Jerry <fought> over the zephyr."
    out = render_front(sentence, "zephyr", [])
    assert "&amp;" in out and "&lt;fought&gt;" in out


def test_front_styles_only_the_matched_occurrence():
    """A word appearing twice must not be styled where it was not matched."""
    sentence = "The zephyr died; another zephyr rose."
    matches = detect_reuse(sentence, ["zephyr"], "x")
    out = render_front(sentence, "", matches)
    assert out.count("<i>zephyr</i>") == 2  # both were matched here


def test_front_never_nests_reuse_inside_the_target():
    sentence = "The glib remark was glib."
    out = render_front(sentence, "glib", detect_reuse(sentence, ["glib"], "glib"))
    assert "<i><u><i>" not in out


def test_front_with_no_reuse_still_marks_the_target():
    out = render_front("The judge was obdurate.", "obdurate", [])
    assert out == "The judge was <i><u>obdurate</u></i>."


# --- ranking over-generated candidates --------------------------------------

from vocab_gen.render import rank_candidates


def c(idx, reused=(), missing=False, giveaway=(), sense=None):
    return {
        "sentence": f"s{idx}", "reused": list(reused), "missing_target": missing,
        "giveaway": list(giveaway), "wrong_sense": sense,
    }


def test_reusing_candidates_come_first():
    out = rank_candidates([c(0), c(1, reused=["x"]), c(2)], keep=2)
    assert out[0]["sentence"] == "s1"


def test_a_candidate_without_the_target_is_ranked_last():
    """Worse than reusing nothing: the card would have no word to underline."""
    out = rank_candidates([c(0, missing=True, reused=["x"]), c(1)], keep=2)
    assert out[0]["sentence"] == "s1"


def test_giveaways_are_demoted_within_a_tier():
    out = rank_candidates([c(0, reused=["x"], giveaway=["y"]), c(1, reused=["x"])], keep=2)
    assert out[0]["sentence"] == "s1"


def test_wrong_sense_outranks_giveaway_as_a_problem():
    out = rank_candidates(
        [c(0, reused=["x"], sense="NOUN"), c(1, reused=["x"], giveaway=["y"])], keep=2
    )
    assert out[0]["sentence"] == "s1"


def test_the_models_order_is_kept_within_a_tier():
    out = rank_candidates([c(0, reused=["x"]), c(1, reused=["x"])], keep=2)
    assert [x["sentence"] for x in out] == ["s0", "s1"]


def test_ranking_never_returns_fewer_than_asked_when_it_has_them():
    """The reason this ranks rather than filters: reuse can be scarce."""
    out = rank_candidates([c(0), c(1), c(2), c(3)], keep=3)
    assert len(out) == 3  # none reuse, but three are still shown


def test_keep_larger_than_the_pool_returns_the_pool():
    assert len(rank_candidates([c(0), c(1)], keep=5)) == 2
