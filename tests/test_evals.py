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
        from vocab_gen.generate import Outcome

        result = types.SimpleNamespace(
            definition=["d"],
            candidates=[cand("The obdurate judge spoke.", "obdurate")],
        )
        usage = types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return Outcome(result, usage, model, "low")

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
        from vocab_gen.generate import Outcome

        result = types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        )
        usage = types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return Outcome(result, usage, model, "low")

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    runner.run(["anthropic:a"], n_cases=1, judge_model=None)
    assert recorded == []


def test_results_round_trip_to_disk(monkeypatch, tmp_path):
    def fake_generate(word, words, n=3, model=None, prefer=None, avoid=None, **kw):
        from vocab_gen.generate import Outcome

        result = types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        )
        usage = types.SimpleNamespace(
            input_tokens=5, output_tokens=7,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return Outcome(result, usage, model, "low")

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    run_id, rows = runner.run(["anthropic:a"], n_cases=1, judge_model=None)
    loaded = runner.load(run_id)
    assert len(loaded) == len(rows)
    assert loaded[0]["word"] == rows[0].word


def test_a_failing_model_does_not_abort_the_run(monkeypatch, tmp_path):
    def fake_generate(word, words, n=3, model=None, **kw):
        if model == "bad":
            raise RuntimeError("no balance")
        from vocab_gen.generate import Outcome

        result = types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        )
        usage = types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return Outcome(result, usage, model, "low")

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    _id, rows = runner.run(["bad", "anthropic:a"], n_cases=1, judge_model=None)
    assert any(r.error for r in rows) and any(not r.error for r in rows)


# --- prompt variants ---------------------------------------------------------


def test_each_variant_actually_reaches_the_model(monkeypatch, tmp_path):
    """The bug this guards: threading the variant is easy to drop, and every
    variant then silently runs the baseline prompt."""
    seen = []

    def fake_generate(word, words, n=3, model=None, variant=None, **kw):
        seen.append(getattr(variant, "name", variant))
        return _fake_outcome(model)

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    runner.run(["m"], variants=["baseline", "terse"], n_cases=2, judge_model=None)
    assert set(seen) == {"baseline", "terse"}
    assert len(seen) == 4  # 2 variants x 2 cases


def test_rows_record_which_variant_produced_them(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "generate", lambda *a, **kw: _fake_outcome("m"))
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    _id, rows = runner.run(["m"], variants=["baseline", "terse"], n_cases=1, judge_model=None)
    assert {r.variant for r in rows} == {"baseline", "terse"}


def test_variants_default_to_baseline_only(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "generate", lambda *a, **kw: _fake_outcome("m"))
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    _id, rows = runner.run(["m"], n_cases=1, judge_model=None)
    assert {r.variant for r in rows} == {"baseline"}


def _fake_outcome(model):
    from vocab_gen.generate import Outcome

    return Outcome(
        types.SimpleNamespace(
            definition=["d"], candidates=[cand("The obdurate judge spoke.", "obdurate")]
        ),
        types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
        model,
        "low",
    )


# --- paired comparison -------------------------------------------------------


def prow(variant, word, judge, **kw):
    base = dict(
        model="m", variant=variant, word=word, register="abstract", index=0,
        sentence="s", words=10, has_target=True, claimed=1, verified=1,
        reused=["x"], no_invented_reuse=True, invented=[], gives_away=False,
        giveaway_words=[], usable=True, latency=1.0, input_tokens=1,
        output_tokens=1, cache_read=0, cost=0.01, error="", naturalness=4,
        sense_fit=4, recall_value=4, reuse_fit=4, judge_note="",
        judge_overall=judge,
    )
    base.update(kw)
    return base


def test_paired_measures_the_per_word_difference():
    rows = [prow("a", "w1", 4.0), prow("b", "w1", 3.0),
            prow("a", "w2", 5.0), prow("b", "w2", 4.0)]
    res = report.paired(rows, "a", "b")
    assert res["mean_diff"] == pytest.approx(1.0)
    assert res["n_words"] == 2 and res["a_wins"] == 2


def test_pairing_is_more_sensitive_than_comparing_averages():
    """The reason this is paired: word difficulty dwarfs the effect being measured."""
    import statistics

    # Words differ hugely in difficulty; the variant is consistently +0.3 better.
    difficulty = [1.0, 5.0, 2.0, 4.5, 1.5, 4.0, 2.5, 3.5]
    rows = []
    for i, d in enumerate(difficulty):
        rows.append(prow("a", f"w{i}", d + 0.3))
        rows.append(prow("b", f"w{i}", d))

    res = report.paired(rows, "a", "b")
    assert res["significant"] is True, "pairing should resolve a consistent effect"

    # Unpaired, the same data is swamped by how much the words differ.
    a = [r["judge_overall"] for r in rows if r["variant"] == "a"]
    b = [r["judge_overall"] for r in rows if r["variant"] == "b"]
    pooled_se = (statistics.stdev(a) ** 2 / len(a) + statistics.stdev(b) ** 2 / len(b)) ** 0.5
    unpaired_t = (statistics.mean(a) - statistics.mean(b)) / pooled_se
    assert abs(unpaired_t) < 1.96, "unpaired should fail to resolve it"


def test_paired_calls_a_wash_a_wash():
    rows = []
    for i, (x, y) in enumerate([(4.0, 4.1), (3.5, 3.4), (4.2, 4.2), (3.8, 3.9)]):
        rows.append(prow("a", f"w{i}", x))
        rows.append(prow("b", f"w{i}", y))
    assert report.paired(rows, "a", "b")["significant"] is False


def test_paired_needs_words_present_in_both_arms():
    rows = [prow("a", "w1", 4.0), prow("b", "w2", 3.0)]
    assert report.paired(rows, "a", "b") is None


def test_paired_ignores_errored_rows():
    rows = [prow("a", "w1", 4.0), prow("b", "w1", 3.0),
            prow("a", "w2", 4.5), prow("b", "w2", 3.5),
            prow("a", "w3", 9.0, error="boom"), prow("b", "w3", 3.0)]
    res = report.paired(rows, "a", "b")
    assert res["n_words"] == 2  # w3 dropped because one arm failed
    assert res["mean_diff"] == pytest.approx(1.0)  # the 9.0 never counted


def test_paired_needs_at_least_two_shared_words():
    """One word gives no variance, so no interval can be computed."""
    rows = [prow("a", "w1", 4.0), prow("b", "w1", 3.0)]
    assert report.paired(rows, "a", "b") is None


def test_paired_can_compare_any_metric():
    rows = [prow("a", "w1", 4.0, verified=2), prow("b", "w1", 4.0, verified=0),
            prow("a", "w2", 4.0, verified=3), prow("b", "w2", 4.0, verified=1)]
    res = report.paired(rows, "a", "b", metric="verified")
    assert res["mean_diff"] == pytest.approx(2.0)


def test_arm_label_keeps_baseline_clean():
    assert report.arm_of(prow("baseline", "w", 4.0)) == "m"
    assert report.arm_of(prow("terse", "w", 4.0)) == "m · terse"


def test_render_paired_reports_each_variant():
    rows = []
    for i in range(4):
        rows.append(prow("baseline", f"w{i}", 3.5))
        rows.append(prow("terse", f"w{i}", 4.0))
    out = report.render_paired(rows)
    assert "terse" in out and "judge" in out


def test_the_original_twenty_remain_a_prefix():
    """Appending keeps subset(20) reproducing the set earlier runs used."""
    first = [c.word for c in subset(20)]
    assert first[0] == "hidebound" and first[-1] == "tour de force"
    assert len(GOLDEN) == 40


def test_the_enlarged_set_is_better_balanced():
    import collections

    reg = collections.Counter(c.register for c in GOLDEN)
    assert reg["concrete"] >= 15, reg
    pos = collections.Counter(c.pos for c in GOLDEN)
    assert min(pos.values()) >= 2, pos


def test_oversampling_can_belong_to_a_variant(monkeypatch, tmp_path):
    """So two arms differing only in over-sampling can be judged in one run."""
    from vocab_gen.prompts import get

    assert get("baseline-os2").oversample == 2
    assert get("baseline").oversample == 1

    asked = []

    def fake_generate(word, words, n=3, oversample=1, **kw):
        asked.append((n, oversample))
        return _fake_outcome("m")

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    runner.run(["m"], variants=["baseline", "baseline-os2"], n_cases=1, judge_model=None)
    assert (3, 1) in asked and (3, 2) in asked


def test_the_variants_own_oversampling_reaches_generate(monkeypatch):
    from vocab_gen import generate as gen
    from vocab_gen.prompts import get

    seen = {}
    monkeypatch.setattr(
        gen, "_generate_once",
        lambda spec, word, words, n, *a, **kw: seen.update(n=n) or gen.Outcome(
            types.SimpleNamespace(definition=[], candidates=[]), None, spec, "low"
        ),
    )
    gen.generate("x", ["y"], n=3, variant=get("baseline-os2"))
    assert seen["n"] == 6, "variant oversampling must reach the request"


def test_oversampled_selection_survives_markdown(monkeypatch, tmp_path):
    """prepare() strips markdown, so selecting by sentence text silently drops
    any candidate the model emphasised."""
    from vocab_gen.generate import Outcome

    def fake_generate(word, words, n=3, oversample=1, **kw):
        cands = [
            cand("The **obdurate** judge spoke first.", "obdurate"),
            cand("The obdurate judge spoke second.", "obdurate"),
            cand("The obdurate judge spoke third.", "obdurate"),
            cand("The obdurate judge spoke fourth.", "obdurate"),
            cand("The obdurate judge spoke fifth.", "obdurate"),
            cand("The obdurate judge spoke sixth.", "obdurate"),
        ]
        return Outcome(
            types.SimpleNamespace(definition=["d"], part_of_speech="adjective",
                                  candidates=cands),
            types.SimpleNamespace(input_tokens=1, output_tokens=1,
                                  cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0),
            "m", "low",
        )

    monkeypatch.setattr(runner, "generate", fake_generate)
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    _id, rows = runner.run(["m"], variants=["baseline-os2"], n_cases=1, judge_model=None)
    assert len(rows) == 3, "all three selected candidates must survive"
