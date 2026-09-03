import json
import random

from vocab_gen.history import History

DECK = [f"word{i:03d}" for i in range(100)]


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
    used = DECK[:10]
    for _ in range(3):
        a.record(used)
    prefer, _ = a.plan(DECK, n_prefer=20)
    assert not (set(prefer) & set(used))


def test_avoid_is_the_most_recently_used(tmp_path):
    a = h(tmp_path)
    a.record(["word000"])
    a.record(["word001"])
    _, avoid = a.plan(DECK, n_avoid=2)
    assert set(avoid) == {"word000", "word001"}


def test_prefer_is_sampled_not_alphabetical(tmp_path):
    """The whole point: with everything unused, don't just take the first N."""
    a = h(tmp_path)
    runs = {tuple(a.plan(DECK, n_prefer=10, rng=random.Random(seed))[0]) for seed in range(5)}
    assert len(runs) > 1, "plan() returned the same slice every time"
    alphabetical = tuple(sorted(DECK)[:10])
    assert alphabetical not in runs or len(runs) > 1


def test_prefer_tops_up_when_pool_is_small(tmp_path):
    a = h(tmp_path)
    small = DECK[:5]
    a.record(small[:2])
    prefer, _ = a.plan(small, n_prefer=4)
    assert len(prefer) == 4  # falls through to the next-least-used tier


def test_plan_ignores_words_no_longer_in_the_deck(tmp_path):
    a = h(tmp_path)
    a.record(["deleted-card-word"])
    prefer, avoid = a.plan(DECK, n_prefer=5, n_avoid=5)
    assert "deleted-card-word" not in avoid
    assert all(w in DECK for w in prefer)


def test_coverage_counts_only_deck_words(tmp_path):
    a = h(tmp_path)
    a.record(["word000", "word001", "gone"])
    a.record(["word000"])
    cov = a.coverage(DECK)
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
