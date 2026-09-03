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

KNOWN = {"zephyr", "strait", "parsimonious", "cinéma vérité", "pied-à-terre"}


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
