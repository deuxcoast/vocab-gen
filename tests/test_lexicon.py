import random

from vocab_gen.collection import VocabWord
from vocab_gen.lexicon import (
    MIN_LISTS,
    MIN_MARKS,
    Band,
    Marks,
    calibration_sample,
    deck_keys,
    deck_report,
    frontier,
    prevalence,
    recent_band,
    suggest,
    word_lists,
)


def marks(tmp_path, known=(), unknown=()):
    m = Marks(tmp_path / "vocabulary.json")
    m.mark(known, known=True)
    m.mark(unknown, known=False)
    return m


# --- the vendored data ---------------------------------------------------


def test_the_word_lists_load_and_carry_consensus_counts():
    lists = word_lists()
    assert len(lists) > 9000
    # A word every prep course teaches should be on many lists; an obscure one few.
    assert lists["ubiquitous"] >= 8
    assert all(n >= 1 for n in lists.values())


def test_prevalence_loads_and_is_on_the_probit_scale():
    prev = prevalence()
    assert len(prev) > 60000
    assert prev["wheelbarrow"] > prev["obdurate"]  # both rare, one universally known


# --- matching against the deck -------------------------------------------


def test_deck_keys_cover_inflections_so_a_word_you_have_is_not_suggested():
    keys = deck_keys([VocabWord("flagons"), VocabWord("descried")])
    assert "flagon" in keys and "descry" in keys


def test_a_word_already_in_the_deck_is_never_suggested(tmp_path):
    deck = [VocabWord("dilettante")]
    assert "dilettante" not in {c.word for c in suggest(deck, marks(tmp_path), n=200)}


def test_a_word_marked_known_is_never_suggested(tmp_path):
    plain = {c.word for c in suggest([], marks(tmp_path), n=200)}
    assert plain, "expected some suggestions to begin with"
    target = next(iter(plain))
    after = {c.word for c in suggest([], marks(tmp_path, known=[target]), n=200)}
    assert target not in after


# --- the frontier --------------------------------------------------------


def test_without_enough_marks_the_band_is_deck_derived_and_says_so(tmp_path):
    deck = [VocabWord("obdurate", created=1.0), VocabWord("zephyr", created=2.0)]
    band = frontier(marks(tmp_path), deck)
    assert band.n_marks < MIN_MARKS
    assert "marks" not in band.source  # must not claim to be measured


def test_enough_marks_fit_a_frontier_between_known_and_unknown(tmp_path):
    """The whole point: population prevalence locates a neighbourhood, the
    learner's own answers locate the edge."""
    prev = prevalence()
    ranked = sorted((w for w in word_lists() if w in prev), key=lambda w: prev[w])
    hard, easy = ranked[:15], ranked[-15:]
    band = frontier(marks(tmp_path, known=easy, unknown=hard), [])
    assert band.source == "fitted from your marks"
    assert band.n_marks >= MIN_MARKS
    assert prev[hard[-1]] <= band.ceiling <= prev[easy[0]]


def test_recent_additions_drive_the_fallback_not_the_whole_deck():
    """A deck records years of history; the frontier is where the learner is now."""
    prev = prevalence()
    easy = [w for w in prev if prev[w] > 2.0][:40]
    hard = [w for w in prev if -0.5 < prev[w] < 0.0][:40]
    deck = ([VocabWord(w, created=1_000.0) for w in easy]      # long ago
            + [VocabWord(w, created=2_000_000.0) for w in hard])  # lately
    band = recent_band(deck)
    assert band.ceiling < 1.0, "the old easy words must not set today's ceiling"


def test_a_deck_with_no_history_falls_back_without_crashing():
    band = recent_band([VocabWord("obdurate")])
    assert band.floor < band.ceiling and "default" in band.source


# --- calibration ---------------------------------------------------------


def test_the_calibration_sample_spans_both_sides_of_the_frontier():
    prev = prevalence()
    got = calibration_sample([], n=30, rng=random.Random(0))
    vals = [prev[w] for w in got]
    assert len(got) == 30
    assert min(vals) < 0 < max(vals), "a one-sided sample locates nothing"


def test_calibration_does_not_waste_questions_on_certainties():
    """Asking about `dent` spends attention confirming what is not in doubt."""
    prev = prevalence()
    got = calibration_sample([], n=40, rng=random.Random(0))
    assert max(prev[w] for w in got) <= 1.5


def test_calibration_never_asks_about_words_already_in_the_deck():
    got = calibration_sample([VocabWord("quotidian")], n=40, rng=random.Random(0))
    assert "quotidian" not in got


# --- marks store ---------------------------------------------------------


def test_marks_round_trip(tmp_path):
    m = marks(tmp_path, known=["alacrity"], unknown=["obloquy"])
    m.save()
    again = Marks.load(tmp_path / "vocabulary.json")
    assert "alacrity" in again.known and "obloquy" in again.unknown


def test_a_later_answer_replaces_an_earlier_one(tmp_path):
    m = marks(tmp_path, unknown=["alacrity"])
    m.mark(["alacrity"], known=True)
    assert "alacrity" in m.known and "alacrity" not in m.unknown


def test_a_missing_or_corrupt_store_is_not_an_error(tmp_path):
    assert len(Marks.load(tmp_path / "nope.json")) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert len(Marks.load(bad)) == 0


# --- reporting -----------------------------------------------------------


def test_deck_report_counts_list_membership(tmp_path):
    report = deck_report([VocabWord("dilettante"), VocabWord("zzzznotaword")], marks(tmp_path))
    assert report["deck"] == 2
    assert report["on_list"] == 1


def test_suggestions_are_ordered_by_consensus_then_rarity(tmp_path):
    got = suggest([], marks(tmp_path), n=40, band=Band(-1.0, 1.0, "test"))
    counts = [c.n_lists for c in got]
    assert counts == sorted(counts, reverse=True)
    assert all(c.n_lists >= MIN_LISTS for c in got)
