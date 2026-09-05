import json
import random

from vocab_gen.collection import VocabWord
from vocab_gen.history import History

DECK = [VocabWord(f"word{i:03d}") for i in range(100)]
TERMS = [w.term for w in DECK]


def h(tmp_path):
    return History(tmp_path / "usage.json")


def test_record_and_roundtrip(tmp_path):
    a = h(tmp_path)
    a.record(["Frigate", "grove"])
    a.record(["frigate"])
    a.save()

    b = History.load(tmp_path / "usage.json")
    assert b.words["frigate"]["n"] == 2  # case-insensitive
    assert b.words["grove"]["n"] == 1


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert History.load(tmp_path / "nope.json").words == {}


def test_corrupt_file_does_not_crash(tmp_path):
    p = tmp_path / "usage.json"
    p.write_text("{not json")
    assert History.load(p).words == {}


def test_wrong_version_is_discarded(tmp_path):
    p = tmp_path / "usage.json"
    p.write_text(json.dumps({"version": 999, "words": {"x": {"n": 5}}}))
    assert History.load(p).words == {}


def test_prefer_excludes_recently_used(tmp_path):
    a = h(tmp_path)
    used = TERMS[:10]
    for _ in range(3):
        a.record(used)
    prefer, _ = a.plan(DECK, n_prefer=20)
    assert not ({w.term for w in prefer} & set(used))


def test_avoid_is_the_most_recently_used(tmp_path):
    a = h(tmp_path)
    a.record(["word000"])
    a.record(["word001"])
    _, avoid = a.plan(DECK, n_avoid=2)
    assert set(avoid) == {"word000", "word001"}


def test_prefer_is_sampled_not_alphabetical(tmp_path):
    """The whole point: with everything unused, don't just take the first N."""
    a = h(tmp_path)
    runs = {
        tuple(w.term for w in a.plan(DECK, n_prefer=10, rng=random.Random(seed))[0])
        for seed in range(5)
    }
    assert len(runs) > 1, "plan() returned the same slice every time"
    alphabetical = tuple(sorted(TERMS)[:10])
    assert alphabetical not in runs or len(runs) > 1


def test_prefer_tops_up_when_pool_is_small(tmp_path):
    a = h(tmp_path)
    small = DECK[:5]
    a.record([w.term for w in small[:2]])
    prefer, _ = a.plan(small, n_prefer=4)
    assert len(prefer) == 4  # falls through to the next-least-used tier


def test_plan_ignores_words_no_longer_in_the_deck(tmp_path):
    a = h(tmp_path)
    a.record(["deleted-card-word"])
    prefer, avoid = a.plan(DECK, n_prefer=5, n_avoid=5)
    assert "deleted-card-word" not in avoid
    assert all(w.term in TERMS for w in prefer)


def test_coverage_counts_only_deck_words(tmp_path):
    a = h(tmp_path)
    a.record(["word000", "word001", "gone"])
    a.record(["word000"])
    cov = a.coverage(TERMS)
    assert cov["deck"] == 100
    assert cov["seen"] == 2 and cov["unseen"] == 98
    assert cov["uses"] == 3
    assert cov["top"][0] == ("word000", 2)


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    a = h(tmp_path)
    a.record(["word000"])
    a.save()
    a.save()
    assert [p.name for p in tmp_path.iterdir()] == ["usage.json"]


# --- shakiness weighting ----------------------------------------------------


def test_prefer_favours_words_you_keep_forgetting(tmp_path):
    """A lapsed word should surface far more often than a solid one."""
    deck = [VocabWord("solid", ivl=200) for _ in range(0)]
    deck = [VocabWord(f"solid{i}", ivl=200) for i in range(20)]
    deck.append(VocabWord("gainsay", lapses=5, ivl=8))
    a = h(tmp_path)

    picked = sum(
        "gainsay" in {w.term for w in a.plan(deck, n_prefer=3, rng=random.Random(s))[0]}
        for s in range(200)
    )
    # Uniform sampling would pick it ~3/21 = 14% of the time.
    assert picked > 60, f"shaky word picked only {picked}/200 times"


def test_shakiness_ranks_lapses_above_short_intervals():
    assert VocabWord("a", lapses=3).shakiness > VocabWord("b", ivl=5).shakiness
    assert VocabWord("b", ivl=5).shakiness > VocabWord("c", ivl=365).shakiness


# --- kept candidates --------------------------------------------------------


def test_kept_roundtrips(tmp_path):
    a = h(tmp_path)
    a.record_kept("obdurate", "The chief remained obdurate.", ["concession"])
    a.save()
    b = History.load(tmp_path / "usage.json")
    assert b.recent_kept()[0]["target"] == "obdurate"
    assert b.recent_kept()[0]["reused"] == ["concession"]


def test_kept_is_capped(tmp_path):
    from vocab_gen.history import KEEP_LIMIT

    a = h(tmp_path)
    for i in range(KEEP_LIMIT + 15):
        a.record_kept(f"w{i}", f"sentence {i}", [])
    a.save()
    b = History.load(tmp_path / "usage.json")
    assert len(b.kept) == KEEP_LIMIT
    assert b.kept[-1]["target"] == f"w{KEEP_LIMIT + 14}"  # newest survive


def test_recent_kept_returns_the_newest(tmp_path):
    a = h(tmp_path)
    for i in range(6):
        a.record_kept(f"w{i}", f"s{i}", [])
    assert [k["target"] for k in a.recent_kept(2)] == ["w4", "w5"]


# --- FSRS-weighted selection -------------------------------------------------


def test_selection_favours_words_the_learner_is_likely_to_have_forgotten(tmp_path):
    """Weighting now comes from a memory model rather than a hand-rolled score."""
    import time

    fresh = [
        VocabWord(f"solid{i}", stability=400.0, decay=0.135, last_review=time.time())
        for i in range(20)
    ]
    # One word last seen a full stability ago: recall is 90%, so 10% forgotten.
    due = VocabWord(
        "atavism", stability=10.0, decay=0.135, last_review=time.time() - 10 * 86400
    )
    deck = fresh + [due]
    a = h(tmp_path)

    picked = sum(
        "atavism" in {w.term for w in a.plan(deck, n_prefer=3, rng=random.Random(s))[0]}
        for s in range(200)
    )
    # Uniform sampling would pick it about 3/21 = 14% of the time.
    assert picked > 60, f"forgotten word picked only {picked}/200 times"


def test_the_three_weightings_choose_differently(tmp_path):
    import time

    now = time.time()
    deck = [
        # Freshly reviewed but heavily lapsed: the old score calls this shaky,
        # FSRS knows it was just seen and is not.
        VocabWord("recent", lapses=5, ivl=10, stability=200.0, decay=0.135, last_review=now),
        # Never lapsed but long overdue: FSRS sees the risk, the old score cannot.
        VocabWord("overdue", lapses=0, ivl=10, stability=5.0, decay=0.135,
                  last_review=now - 60 * 86400),
    ]
    a = h(tmp_path)
    assert deck[1].shakiness > deck[0].shakiness, "FSRS should rank the overdue word shakier"
    assert deck[0].legacy_shakiness > deck[1].legacy_shakiness, "the old score ranks it backwards"


def test_uniform_weighting_ignores_memory_entirely(tmp_path):
    import time

    deck = [
        VocabWord("a", stability=5.0, decay=0.135, last_review=time.time() - 60 * 86400),
        VocabWord("b", stability=400.0, decay=0.135, last_review=time.time()),
    ]
    a = h(tmp_path)
    picks = [
        a.plan(deck, n_prefer=1, rng=random.Random(s), weighting="uniform")[0][0].term
        for s in range(60)
    ]
    assert len(set(picks)) == 2, "uniform must not favour either word"


def test_fsrs_weighting_favours_the_overdue_word(tmp_path):
    import time

    deck = [
        VocabWord("overdue", stability=5.0, decay=0.135, last_review=time.time() - 60 * 86400),
        VocabWord("fresh", stability=400.0, decay=0.135, last_review=time.time()),
    ]
    a = h(tmp_path)
    picks = [
        a.plan(deck, n_prefer=1, rng=random.Random(s), weighting="fsrs")[0][0].term
        for s in range(60)
    ]
    assert picks.count("overdue") > 55, f"only {picks.count('overdue')}/60"


def test_an_unknown_weighting_falls_back_to_fsrs(tmp_path):
    deck = [VocabWord("a"), VocabWord("b")]
    assert h(tmp_path).plan(deck, n_prefer=1, weighting="nonsense")[0]


# --- the target is never offered back to the learner -------------------------


def test_the_target_is_held_out_of_its_own_preferred_list(tmp_path):
    """A deck can contain the word you are making a new card for."""
    deck = DECK + [VocabWord("flagons", "A large bottle for wine.")]
    for seed in range(20):
        prefer, _ = h(tmp_path).plan(
            deck, n_prefer=len(deck), rng=random.Random(seed), target="flagon"
        )
        assert "flagons" not in {w.term for w in prefer}, seed


def test_holdout_is_inflection_aware_not_string_equality(tmp_path):
    from vocab_gen.morphology import same_term

    assert same_term("flagons", "flagon")
    assert same_term("descried", "descry")
    assert same_term("prevaricating", "prevaricate")
    # Derivation must not collapse, or unrelated words get held out too.
    assert not same_term("straitjacket", "strait")
    # A phrase cannot collapse onto a single token.
    assert not same_term("de facto", "facto")


def test_without_a_target_nothing_is_held_out(tmp_path):
    deck = DECK + [VocabWord("flagons")]
    prefer, _ = h(tmp_path).plan(deck, n_prefer=len(deck), rng=random.Random(0))
    assert "flagons" in {w.term for w in prefer}
