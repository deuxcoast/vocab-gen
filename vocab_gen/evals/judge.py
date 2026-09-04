"""LLM-as-judge for the part no regex can score: does this read as written prose?

Two deliberate choices.

Blind: the judge never sees which model produced which sentence. Candidates for
one target word are pooled across models, shuffled, and labelled A, B, C.

Comparative: all candidates for a word are scored in a single call, side by
side. Listwise judgement is more stable than scoring sentences in isolation,
and it costs a fraction as much.

A caveat that cannot be engineered away: judging Claude output with a Claude
judge risks self-preference. The judge model is configurable for that reason —
run it twice with judges from different families and compare the ranking before
trusting a close result.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, Field

from ..generate import resolve_model, split_spec
from ..providers import SystemBlock, build

DEFAULT_JUDGE = "claude-opus-5"


class Verdict(BaseModel):
    label: str = Field(description="The candidate's letter label, e.g. 'A'.")
    naturalness: int = Field(
        ge=1, le=5,
        description=(
            "5 = indistinguishable from published prose. 1 = obviously written to "
            "showcase vocabulary. This is the most important axis."
        ),
    )
    sense_fit: int = Field(
        ge=1, le=5, description="Is the target word used in a correct, recognised sense?"
    )
    recall_value: int = Field(
        ge=1, le=5,
        description=(
            "5 = context hints at the meaning but does not hand it over. "
            "1 = the sentence defines the word, so it cannot test recall."
        ),
    )
    reuse_fit: int = Field(
        ge=1, le=5,
        description=(
            "If the sentence reuses other advanced vocabulary, does it belong to the "
            "same subject and register, or was it bolted on? Use 3 when there is no "
            "reuse to judge."
        ),
    )
    note: str = Field(description="One short clause naming the deciding flaw or merit.")


class JudgeResult(BaseModel):
    verdicts: list[Verdict]


RUBRIC = """\
You are grading example sentences written for a native English speaker's personal \
vocabulary deck. Each card shows one sentence containing a target word; the learner \
must infer the meaning, then check it against their own definition on the back.

You will see several candidate sentences for the same target word, from different \
sources, in random order. Score each independently on its own merits.

What matters, in order:

1. It must read like prose someone would actually publish — journalism, criticism, \
fiction, popular history. A sentence whose evident purpose is to display a hard word \
has failed, however grammatical it is.
2. The target word must be used in a genuine sense of that word.
3. The sentence must not give the meaning away. Context should narrow it, not define \
it. An appositive gloss is the worst case.
4. Where other advanced vocabulary appears, it must belong to the same subject and \
register. Two hard words bolted together is a failure even when both are used \
correctly.

Be discriminating. Most sentences are competent; reserve 5 for ones you would be glad \
to find in print, and use the low end when a sentence reads as an exercise. Do not \
reward length, ornament, or rare words for their own sake."""


def judge_case(
    word: str,
    candidates: list[dict],
    model: str = DEFAULT_JUDGE,
    rng: random.Random | None = None,
) -> dict[str, Verdict]:
    """Score candidates for one word. Returns {candidate_id: Verdict}.

    `candidates` are dicts with at least `id` and `sentence`.
    """
    if not candidates:
        return {}
    rng = rng or random.Random()
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    labels = {}
    lines = []
    for i, cand in enumerate(shuffled):
        label = chr(ord("A") + i)
        labels[label] = cand["id"]
        lines.append(f"{label}. {cand['sentence']}")

    provider_name, model_id = split_spec(resolve_model(model))
    provider = build(provider_name)
    completion = provider.complete(
        model=model_id,
        system=[SystemBlock(RUBRIC, cacheable=True)],
        user=(
            f"Target word: {word}\n\n"
            f"Candidates:\n" + "\n".join(lines) + "\n\n"
            "Score every candidate. Use each label exactly once."
        ),
        schema_model=JudgeResult,
        max_tokens=8000,
        effort="low",
    )
    if completion.parsed is None:
        return {}
    return {
        labels[v.label]: v for v in completion.parsed.verdicts if v.label in labels
    }


def overall(v: Verdict) -> float:
    """Naturalness is weighted double — it is the whole point of the tool."""
    return (2 * v.naturalness + v.sense_fit + v.recall_value + v.reuse_fit) / 5
