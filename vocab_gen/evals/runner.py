"""Run the golden set across models and record everything.

Two properties the results depend on:

Identical prompts. The preferred and avoided word lists normally come from usage
history and drift between calls, which would mean each model saw a different
prompt and the comparison measured nothing. They are computed once per run from
a fixed seed and reused for every model and case.

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
from .cases import Case, subset
from .graders import grade_candidate
from .judge import DEFAULT_JUDGE, judge_case, overall

RUNS_DIR = Path("evals/runs")


@dataclass
class Row:
    run_id: str
    model: str
    word: str
    pos: str
    register: str
    index: int
    sentence: str
    words: int
    has_target: bool
    claimed: int
    verified: int
    reused: list
    no_invented_reuse: bool
    invented: list
    gives_away: bool
    giveaway_words: list
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
    n_cases: int | None = None,
    n_candidates: int = 3,
    seed: int = 20260904,
    judge_model: str | None = DEFAULT_JUDGE,
    deck: str = "General",
    label: str = "",
    on_event=lambda *_: None,
) -> tuple[str, list[Row]]:
    vocab = extract_vocab(deck=deck)
    words = [w.term for w in vocab]

    # Frozen once: every model must see the same prompt or nothing is comparable.
    prefer, avoid = History.load().plan(vocab, rng=random.Random(seed))

    cases = subset(n_cases)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"-{label}" if label else "")
    rows: list[Row] = []

    for model in models:
        spec = resolve_model(model)
        for case in cases:
            started = time.perf_counter()
            try:
                outcome = generate(
                    case.word, words, n=n_candidates, model=spec,
                    prefer=prefer, avoid=avoid, kept=[], allow_fallback=False,
                )
                result, usage, used = outcome.result, outcome.usage, outcome.model
            except Exception as exc:
                on_event("error", spec, case.word, str(exc).splitlines()[0])
                rows.append(_error_row(run_id, spec, case, str(exc).splitlines()[0]))
                continue
            latency = time.perf_counter() - started
            cost = cost_of(used, usage)

            for i, cand in enumerate(result.candidates):
                g = grade_candidate(cand, case.word, result.definition, words)
                rows.append(
                    Row(
                        run_id=run_id, model=used, word=case.word, pos=case.pos,
                        register=case.register, index=i,
                        latency=latency, input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_read=usage.cache_read_input_tokens,
                        # generation cost, split across its candidates
                        cost=(cost / len(result.candidates)) if cost is not None else None,
                        **{k: g[k] for k in (
                            "sentence", "words", "has_target", "claimed", "verified",
                            "reused", "no_invented_reuse", "invented", "gives_away",
                            "giveaway_words", "usable")},
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
            {"id": f"{r.model}#{r.word}#{r.index}", "sentence": r.sentence} for r in group
        ]
        try:
            verdicts = judge_case(word, candidates, judge_model, random.Random(seed))
        except Exception as exc:
            on_event("error", judge_model, word, f"judge failed: {exc}")
            continue
        for row in group:
            v = verdicts.get(f"{row.model}#{row.word}#{row.index}")
            if v is None:
                continue
            row.naturalness = v.naturalness
            row.sense_fit = v.sense_fit
            row.recall_value = v.recall_value
            row.reuse_fit = v.reuse_fit
            row.judge_note = v.note
            row.judge_overall = overall(v)
        on_event("judged", judge_model, word, f"{len(verdicts)} scored")


def _error_row(run_id: str, model: str, case: Case, error: str) -> Row:
    return Row(
        run_id=run_id, model=model, word=case.word, pos=case.pos,
        register=case.register, index=0, sentence="", words=0, has_target=False,
        claimed=0, verified=0, reused=[], no_invented_reuse=True, invented=[],
        gives_away=False, giveaway_words=[], usable=False, latency=0.0,
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
