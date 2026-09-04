import random
import types

import pytest

from vocab_gen.evals import report, runner
from vocab_gen.evals.cases import GOLDEN, subset
from vocab_gen.evals.graders import grade_candidate, summarise
from vocab_gen.evals.judge import Verdict, judge_case, overall

DECK = ["scurrilous", "strait", "supplicants"]


def cand(sentence, surface="obdurate", reused=()):
    return types.SimpleNamespace(
        sentence=sentence, surface_form=surface, reused=list(reused)
    )


# --- golden set --------------------------------------------------------------


def test_golden_set_has_no_duplicates():
    words = [c.word for c in GOLDEN]
    assert len(words) == len(set(words))


def test_golden_set_covers_both_registers():
    assert {c.register for c in GOLDEN} == {"concrete", "abstract"}


def test_subset_is_a_prefix_so_small_runs_stay_comparable():
    assert subset(5) == GOLDEN[:5]


# --- programmatic grading ----------------------------------------------------


def test_grade_flags_missing_target():
    g = grade_candidate(cand("He was stubborn and unyielding."), "obdurate", ["Unyielding."], DECK)
    assert g["has_target"] is False
    assert g["usable"] is False


def test_grade_credits_real_reuse():
    g = grade_candidate(
        cand("The obdurate lawyer made a scurrilous attack.", reused=["scurrilous"]),
        "obdurate", ["Unyielding."], DECK,
    )
    assert g["verified"] == 1 and g["usable"] is True


def test_grade_rejects_invented_reuse():
    g = grade_candidate(
        cand("The obdurate judge spoke.", reused=["mawkishness"]),
        "obdurate", ["Unyielding."], DECK,
    )
    assert g["invented"] == ["mawkishness"]
    assert g["usable"] is False


def test_grade_detects_a_giveaway():
    g = grade_candidate(
        cand("Unyielding and obdurate, he refused."), "obdurate", ["Unyielding."], DECK
    )
    assert g["gives_away"] is True
    # A giveaway is a warning, not a hard gate — short overlaps are often innocent.
    assert g["usable"] is True


def test_summarise_reports_rates():
    rows = [
        grade_candidate(cand("The obdurate judge spoke."), "obdurate", ["Unyielding."], DECK),
        grade_candidate(cand("He was merely stubborn."), "obdurate", ["Unyielding."], DECK),
    ]
    s = summarise(rows)
    assert s["n"] == 2 and s["has_target_rate"] == 0.5


def test_summarise_handles_no_rows():
    assert summarise([])["n"] == 0


# --- judging -----------------------------------------------------------------


def test_overall_weights_naturalness_double():
    """A point of naturalness must outweigh a point on any other axis."""
    base = dict(label="A", naturalness=3, sense_fit=3, recall_value=3, reuse_fit=3, note="")
    baseline = overall(Verdict(**base))
    gain_natural = overall(Verdict(**{**base, "naturalness": 4})) - baseline
    for axis in ("sense_fit", "recall_value", "reuse_fit"):
        gain_other = overall(Verdict(**{**base, axis: 4})) - baseline
        assert gain_natural == pytest.approx(2 * gain_other), axis


def test_judge_never_sees_model_identity(monkeypatch):
    """The prompt must carry sentences and labels only."""
    seen = {}

    class FakeProvider:
        def complete(self, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(
                parsed=types.SimpleNamespace(
                    verdicts=[
                        Verdict(label="A", naturalness=4, sense_fit=4,
                                recall_value=4, reuse_fit=3, note="fine")
                    ]
                )
            )

    monkeypatch.setattr("vocab_gen.evals.judge.build", lambda _n: FakeProvider())
    out = judge_case(
        "obdurate",
        [{"id": "claude-sonnet-5#obdurate#0", "sentence": "The judge was obdurate."}],
        rng=random.Random(0),
    )
    assert "claude" not in seen["user"].lower()
    assert "sonnet" not in seen["user"].lower()
    assert out["claude-sonnet-5#obdurate#0"].naturalness == 4


def test_judge_returns_nothing_when_output_is_unparseable(monkeypatch):
    class FakeProvider:
        def complete(self, **kwargs):
            return types.SimpleNamespace(parsed=None)

    monkeypatch.setattr("vocab_gen.evals.judge.build", lambda _n: FakeProvider())
    assert judge_case("x", [{"id": "a", "sentence": "s"}]) == {}


def test_judge_handles_an_empty_candidate_list():
    assert judge_case("x", []) == {}


# --- reporting ---------------------------------------------------------------


def row(model="m", usable=True, judge=4.0, cost=0.01, error="", **kw):
    base = dict(
        model=model, word="w", register="abstract", index=0, sentence="s", words=10,
        has_target=True, claimed=1, verified=1, reused=["x"], no_invented_reuse=True,
        invented=[], gives_away=False, giveaway_words=[], usable=usable,
        latency=1.0, input_tokens=1, output_tokens=1, cache_read=0, cost=cost,
        error=error, naturalness=4, sense_fit=4, recall_value=4, reuse_fit=4,
        judge_note="", judge_overall=judge,
    )
    base.update(kw)
    return base


def test_accepted_requires_usable_and_a_good_score():
    assert report.accepted(row(judge=4.0)) is True
    assert report.accepted(row(judge=2.0)) is False
    assert report.accepted(row(usable=False, judge=5.0)) is False


def test_accepted_without_a_judge_falls_back_to_usable():
    assert report.accepted(row(judge=None)) is True


def test_cost_per_accepted_penalises_a_cheap_model_that_misses():
    cheap = [row(model="cheap", cost=0.001, judge=2.0)] * 4 + [row(model="cheap", cost=0.001)]
    dear = [row(model="dear", cost=0.01)] * 5
    stats = report.by_model(cheap + dear)
    # cheap: 5 candidates, 1 accepted, $0.005 spend -> $0.005 each
    # dear:  5 candidates, 5 accepted, $0.05 spend  -> $0.01 each
    assert stats["cheap"]["cost_per_accepted"] == pytest.approx(0.005)
    assert stats["dear"]["cost_per_accepted"] == pytest.approx(0.01)
    assert stats["cheap"]["accept_rate"] == pytest.approx(0.2)


def test_errors_are_counted_but_excluded_from_rates():
    stats = report.by_model([row(), row(error="boom", usable=False)])
    assert stats["m"]["errors"] == 1
    assert stats["m"]["candidates"] == 1
    assert stats["m"]["usable_rate"] == 1.0


def test_render_survives_missing_judge_scores():
    out = report.render([row(judge=None, naturalness=None)])
    assert "model" in out


def test_render_reports_nothing_gracefully():
    assert report.render([]) == "no results"


# --- run invariants ----------------------------------------------------------


def test_every_model_sees_an_identical_prompt(monkeypatch, tmp_path):
    """If prefer/avoid drift between models the comparison measures nothing."""
    seen = []

    def fake_generate(word, words, n=3, model=None, prefer=None, avoid=None, **kw):
        seen.append((model, tuple(w.term for w in prefer), tuple(avoid)))
        result = types.SimpleNamespace(
            definition=["d"],
            candidates=[cand("The obdurate judge spoke.", "obdurate")],
        )
        usage = types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return result, usage, model, "low"

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    runner.run(["anthropic:a", "other:b"], n_cases=2, judge_model=None)

    prompts = {(p, a) for _m, p, a in seen}
    assert len(prompts) == 1, "prefer/avoid differed between models or cases"


def test_a_run_does_not_pollute_usage_history(monkeypatch, tmp_path):
    """Evaluating the tool must not change the tool's future behaviour."""
    from vocab_gen.history import History

    recorded = []
    monkeypatch.setattr(History, "record", lambda self, used: recorded.append(used))
    monkeypatch.setattr(History, "save", lambda self: recorded.append("saved"))

    def fake_generate(word, words, n=3, model=None, prefer=None, avoid=None, **kw):
        result = types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        )
        usage = types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return result, usage, model, "low"

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    runner.run(["anthropic:a"], n_cases=1, judge_model=None)
    assert recorded == []


def test_results_round_trip_to_disk(monkeypatch, tmp_path):
    def fake_generate(word, words, n=3, model=None, prefer=None, avoid=None, **kw):
        result = types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        )
        usage = types.SimpleNamespace(
            input_tokens=5, output_tokens=7,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return result, usage, model, "low"

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    run_id, rows = runner.run(["anthropic:a"], n_cases=1, judge_model=None)
    loaded = runner.load(run_id)
    assert len(loaded) == len(rows)
    assert loaded[0]["word"] == rows[0].word


def test_a_failing_model_does_not_abort_the_run(monkeypatch, tmp_path):
    def fake_generate(word, words, n=3, model=None, **kw):
        if model == "bad":
            raise SystemExit("no balance")
        result = types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        )
        usage = types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return result, usage, model, "low"

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    _id, rows = runner.run(["bad", "anthropic:a"], n_cases=1, judge_model=None)
    assert any(r.error for r in rows) and any(not r.error for r in rows)
