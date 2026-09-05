"""Run the golden set across models and record everything.

Two properties the results depend on:

Identical prompts. The preferred and avoided word lists normally come from usage
history and drift between calls, which would mean each model saw a different
prompt and the comparison measured nothing. They are computed from a fixed seed
and memoised per (weighting, target), so every arm sharing a weighting sees a
byte-identical prompt for a given word. The lists vary by target only because
the target itself is held out of them.

No side effects. A run never records to usage history; otherwise evaluating a
model would change the behaviour of the tool you are evaluating.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..collection import extract_vocab
from ..generate import generate, resolve_model
from ..history import History
from ..providers import cost_of
from ..prompts import get as get_variant
from .cases import Case, subset
from ..render import prepare, rank_candidates
from .graders import grade_candidate
from .judge import DEFAULT_JUDGE, judge_case, overall

RUNS_DIR = Path("evals/runs")


@dataclass
class Row:
    run_id: str
    model: str
    variant: str
    word: str
    pos: str
    register: str
    index: int
    sentence: str
    words: int
    has_target: bool
    claimed: int
    verified: int
    unreported: int
    targeting: float | None
    reused: list
    no_invented_reuse: bool
    invented: list
    gives_away: bool | None
    giveaway_words: list
    wrong_sense: str
    usable: bool
    # per-generation, repeated on each of its candidates
    latency: float
    input_tokens: int
    output_tokens: int
    cache_read: int
    cost: float | None
    error: str = ""
    naturalness: int | None = None
    sense_fit: int | None = None
    recall_value: int | None = None
    reuse_fit: int | None = None
    judge_note: str = ""
    judge_overall: float | None = None


def run(
    models: list[str],
    variants: list[str] | None = None,
    n_cases: int | None = None,
    n_candidates: int = 3,
    oversample: int = 1,
    seed: int = 20260904,
    judge_model: str | None = DEFAULT_JUDGE,
    deck: str = "General",
    label: str = "",
    on_event=lambda *_: None,
) -> tuple[str, list[Row]]:
    vocab = extract_vocab(deck=deck)
    words = [w.term for w in vocab]
    # Forgetting probability per word, so reuse can be scored on targeting.
    memory = {w.term.lower(): w.shakiness for w in vocab}

    # Frozen per (weighting, target). Arms that differ in how words are chosen
    # must differ in their preferred list — that list *is* the treatment — but
    # the seed is shared, so the only difference is the weighting and not the
    # draw. The target varies the list only by holding the target itself out.
    history = History.load()
    plans: dict[tuple[str, str], tuple] = {}

    def plan_for(weighting: str, target: str):
        """Memoised so arms sharing a weighting get the identical list per word."""
        if (weighting, target) not in plans:
            plans[(weighting, target)] = history.plan(
                vocab, rng=random.Random(seed), weighting=weighting, target=target
            )
        return plans[(weighting, target)]

    cases = subset(n_cases)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"-{label}" if label else "")
    rows: list[Row] = []

    for model in models:
        spec = resolve_model(model)
        for variant_name in (variants or ["baseline"]):
            variant = get_variant(variant_name)
            # The variant may carry its own over-sampling; a run-level flag
            # still works and wins, for a quick sweep without naming a variant.
            factor = max(oversample, getattr(variant, "oversample", 1))
            rounds = getattr(variant, "batches", 1)
            arm = f"{variant_name}+os{factor}" if factor > 1 and factor != variant.oversample else variant_name
            for case in cases:
                prefer, avoid = plan_for(
                    getattr(variant, "weighting", "fsrs"), case.word
                )
                started = time.perf_counter()
                try:
                    outcome = generate(
                        case.word, words, n=n_candidates, model=spec,
                        prefer=prefer, avoid=avoid, kept=[], allow_fallback=False,
                        variant=variant, oversample=factor, batches=rounds,
                    )
                    result, usage, used = outcome.result, outcome.usage, outcome.model
                except Exception as exc:
                    on_event("error", spec, case.word, str(exc).splitlines()[0])
                    rows.append(
                        _error_row(run_id, spec, arm, case, str(exc).splitlines()[0])
                    )
                    continue
                latency = time.perf_counter() - started
                cost = cost_of(used, usage)

                # Rank and truncate exactly as the CLI does, so the eval scores
                # what a user would actually be shown rather than the raw pool.
                shown = result.candidates
                if factor > 1 or rounds > 1:
                    # Match on position, not on sentence text: prepare() strips
                    # markdown, so comparing strings silently fails for any
                    # candidate the model emphasised, and drops it from the run.
                    prepared = prepare(result, case.word, words)
                    order = {id(p): i for i, p in enumerate(prepared)}
                    keep = [
                        order[id(p)]
                        for p in rank_candidates(prepared, n_candidates)
                    ]
                    shown = [result.candidates[i] for i in sorted(keep)]

                for i, cand in enumerate(shown):
                    g = grade_candidate(
                            cand, case.word, result.definition, words,
                            getattr(result, 'part_of_speech', ''), memory,
                        )
                    rows.append(
                        Row(
                            run_id=run_id, model=used, variant=arm,
                            word=case.word, pos=case.pos,
                            register=case.register, index=i,
                            latency=latency, input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_read=usage.cache_read_input_tokens,
                            # generation cost, split across its candidates
                            cost=(cost / max(len(shown), 1)) if cost is not None else None,
                            **{k: g[k] for k in (
                                "sentence", "words", "has_target", "claimed", "verified",
                                "unreported", "targeting", "reused", "no_invented_reuse", "invented", "gives_away",
                                "giveaway_words", "wrong_sense", "usable")},
                        )
                    )
                on_event("done", spec, case.word, f"{latency:.1f}s")

    if judge_model:
        _judge_all(rows, judge_model, seed, on_event)

    _write(run_id, rows)
    return run_id, rows


def _judge_all(rows: list[Row], judge_model: str, seed: int, on_event) -> None:
    by_word: dict[str, list[Row]] = {}
    for row in rows:
        if row.sentence and not row.error:
            by_word.setdefault(row.word, []).append(row)

    for word, group in by_word.items():
        # Pooled across models and shuffled, so the judge cannot tell them apart.
        candidates = [
            {"id": f"{r.model}#{r.variant}#{r.word}#{r.index}", "sentence": r.sentence}
            for r in group
        ]
        try:
            verdicts = judge_case(word, candidates, judge_model, random.Random(seed))
        except Exception as exc:
            on_event("error", judge_model, word, f"judge failed: {exc}")
            continue
        for row in group:
            v = verdicts.get(f"{row.model}#{row.variant}#{row.word}#{row.index}")
            if v is None:
                continue
            row.naturalness = v.naturalness
            row.sense_fit = v.sense_fit
            row.recall_value = v.recall_value
            row.reuse_fit = v.reuse_fit
            row.judge_note = v.note
            row.judge_overall = overall(v)
        on_event("judged", judge_model, word, f"{len(verdicts)} scored")


def _error_row(run_id: str, model: str, variant: str, case: Case, error: str) -> Row:
    return Row(
        run_id=run_id, model=model, variant=variant, word=case.word, pos=case.pos,
        register=case.register, index=0, sentence="", words=0, has_target=False,
        claimed=0, verified=0, unreported=0, targeting=None, reused=[], no_invented_reuse=True, invented=[],
        gives_away=False, giveaway_words=[], wrong_sense="", usable=False, latency=0.0,
        input_tokens=0, output_tokens=0, cache_read=0, cost=None, error=error,
    )


def _write(run_id: str, rows: list[Row]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{run_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return path


def load(run_id: str) -> list[dict]:
    path = RUNS_DIR / f"{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
