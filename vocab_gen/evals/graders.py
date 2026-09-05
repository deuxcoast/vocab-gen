"""Programmatic grading — everything measurable without asking a model.

These reuse the app's own checks rather than reimplementing them, so a grader
can never disagree with what the tool actually does to a card.
"""

from __future__ import annotations

from ..morphology import match_span
from ..render import (
    detect_reuse,
    gives_away_answer,
    strip_markdown,
    unknown_claims,
    wrong_sense,
)


# A candidate failing any of these is unusable regardless of how it reads.
HARD_GATES = ("has_target", "no_invented_reuse")


def grade_candidate(
    candidate,
    word: str,
    definition: list[str],
    deck: list[str],
    pos: str = "",
    memory: dict | None = None,
) -> dict:
    sentence = strip_markdown(candidate.sentence)
    surface = strip_markdown(candidate.surface_form)
    # Detected, not claimed: a model reuses words it does not report, and
    # measuring its self-report understates what the sentence actually does.
    matches = detect_reuse(sentence, deck, word)
    verified = [m.term for m in matches]
    invented = unknown_claims(candidate.reused, deck)
    giveaway = gives_away_answer(sentence, definition, word)
    present = match_span(sentence, word) or match_span(sentence, surface)
    sense = wrong_sense(sentence, word, pos)

    # How well-targeted the reuse was: the mean probability that the learner had
    # forgotten the words this sentence brought back. Without this, changing how
    # words are chosen cannot be measured at all — the other metrics only see
    # what happened to the sentence.
    targeting = None
    if memory and verified:
        scores = [memory[t.lower()] for t in verified if t.lower() in memory]
        targeting = sum(scores) / len(scores) if scores else None

    return {
        "targeting": targeting,
        "sentence": sentence,
        "words": len(sentence.split()),
        "has_target": present is not None,
        "claimed": len(candidate.reused),
        "verified": len(verified),
        "unreported": len(
            [t for t in verified if not any(r.lower() == t.lower() for r in candidate.reused)]
        ),
        "reused": verified,
        "no_invented_reuse": not invented,
        "invented": invented,
        "gives_away": bool(giveaway),
        "wrong_sense": sense or "",
        "giveaway_words": giveaway,
        "usable": present is not None and not invented,
    }


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-candidate grades into rates."""
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "usable_rate": sum(r["usable"] for r in rows) / n,
        "has_target_rate": sum(r["has_target"] for r in rows) / n,
        "reuse_rate": sum(bool(r["verified"]) for r in rows) / n,
        "reuses_per_sentence": sum(r["verified"] for r in rows) / n,
        "invented_rate": sum(bool(r["invented"]) for r in rows) / n,
        "giveaway_rate": sum(r["gives_away"] for r in rows) / n,
        "wrong_sense_rate": sum(bool(r.get("wrong_sense")) for r in rows) / n,
        "mean_words": sum(r["words"] for r in rows) / n,
        "targeting": (
            sum(r["targeting"] for r in rows if r.get("targeting") is not None)
            / max(sum(1 for r in rows if r.get("targeting") is not None), 1)
        ),
    }
