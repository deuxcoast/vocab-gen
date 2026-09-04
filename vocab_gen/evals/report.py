"""Turn a run into a comparison you can act on."""

from __future__ import annotations

import statistics
from collections import defaultdict

# A card is worth keeping only if it clears the hard gates and reads well enough.
ACCEPT_THRESHOLD = 3.5


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def accepted(row: dict) -> bool:
    if not row.get("usable") or row.get("error"):
        return False
    score = row.get("judge_overall")
    return score is None or score >= ACCEPT_THRESHOLD


def by_model(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)

    out = {}
    for model, group in groups.items():
        good = [r for r in group if not r["error"]]
        errors = [r for r in group if r["error"]]
        # Cost is per-candidate; a generation's cost is spread across its candidates,
        # so summing over candidates recovers the true spend.
        spend = sum(r["cost"] or 0 for r in good)
        n_accepted = sum(accepted(r) for r in good)
        # Latency is recorded per generation but repeated on each candidate.
        gens = {(r["word"], r["model"]): r["latency"] for r in good}

        out[model] = {
            "candidates": len(good),
            "errors": len(errors),
            "usable_rate": _mean([r["usable"] for r in good]),
            "has_target_rate": _mean([r["has_target"] for r in good]),
            "reuse_rate": _mean([bool(r["verified"]) for r in good]),
            "reuses_per_sentence": _mean([r["verified"] for r in good]),
            "invented_rate": _mean([bool(r["invented"]) for r in good]),
            "giveaway_rate": _mean([r["gives_away"] for r in good]),
            "naturalness": _mean([r["naturalness"] for r in good]),
            "recall_value": _mean([r["recall_value"] for r in good]),
            "reuse_fit": _mean([r["reuse_fit"] for r in good]),
            "judge_overall": _mean([r["judge_overall"] for r in good]),
            "accept_rate": n_accepted / len(good) if good else 0.0,
            "spend": spend,
            "cost_per_accepted": (spend / n_accepted) if n_accepted else None,
            "median_latency": statistics.median(gens.values()) if gens else None,
            "mean_words": _mean([r["words"] for r in good]),
        }
    return out


def by_register(rows: list[dict]) -> dict[str, dict]:
    """Does reuse hold up on abstract words, or only concrete ones?"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not row["error"]:
            groups[(row["model"], row["register"])].append(row)
    return {
        f"{model}/{register}": {
            "reuse_rate": _mean([bool(r["verified"]) for r in group]),
            "naturalness": _mean([r["naturalness"] for r in group]),
        }
        for (model, register), group in groups.items()
    }


def render(rows: list[dict]) -> str:
    stats = by_model(rows)
    if not stats:
        return "no results"

    lines = []
    head = (
        f"{'model':30s} {'n':>4s} {'err':>4s} {'usable':>7s} {'reuse':>6s} "
        f"{'inv':>5s} {'give':>5s} {'natur':>6s} {'recall':>6s} {'judge':>6s} "
        f"{'acc':>5s} {'secs':>5s} {'$/100 acc':>10s}"
    )
    lines.append(head)
    lines.append("-" * len(head))
    ranked = sorted(
        stats.items(),
        key=lambda kv: (kv[1]["judge_overall"] or 0, kv[1]["accept_rate"]),
        reverse=True,
    )
    for model, s in ranked:
        per100 = f"{s['cost_per_accepted'] * 100:10.3f}" if s["cost_per_accepted"] is not None else "         ?"
        lines.append(
            f"{model:30s} {s['candidates']:4d} {s['errors']:4d} "
            f"{s['usable_rate'] or 0:7.0%} {s['reuse_rate'] or 0:6.0%} "
            f"{s['invented_rate'] or 0:5.0%} {s['giveaway_rate'] or 0:5.0%} "
            f"{s['naturalness'] or 0:6.2f} {s['recall_value'] or 0:6.2f} "
            f"{s['judge_overall'] or 0:6.2f} {s['accept_rate']:5.0%} "
            f"{s['median_latency'] or 0:5.1f} {per100}"
        )

    lines.append("")
    lines.append(
        "usable = has the target word and invents no reuse · reuse = reuses >=1 deck "
        "word · inv = invented a deck word"
    )
    lines.append(
        "give = leaks the definition · natur/recall/judge = blind judge, 1-5 · "
        f"acc = usable and judged >= {ACCEPT_THRESHOLD}"
    )
    lines.append(
        "$/100 acc = dollars per 100 *accepted* cards — a cheap model that needs more "
        "attempts is not cheap"
    )

    reg = by_register(rows)
    if reg:
        lines.append("")
        lines.append("reuse rate by word type (does it hold up on abstract words?)")
        for key in sorted(reg):
            r = reg[key]
            lines.append(
                f"  {key:38s} reuse {r['reuse_rate'] or 0:5.0%}  "
                f"naturalness {r['naturalness'] or 0:.2f}"
            )
    return "\n".join(lines)


def examples(rows: list[dict], n: int = 3) -> str:
    """The best and worst the judge saw, with its reasoning."""
    scored = [r for r in rows if r.get("judge_overall") is not None]
    if not scored:
        return ""
    scored.sort(key=lambda r: r["judge_overall"], reverse=True)
    out = ["", "highest rated"]
    for r in scored[:n]:
        out.append(f"  [{r['judge_overall']:.1f}] {r['model']} · {r['word']}")
        out.append(f"      {r['sentence']}")
        out.append(f"      judge: {r['judge_note']}")
    out.append("")
    out.append("lowest rated")
    for r in scored[-n:]:
        out.append(f"  [{r['judge_overall']:.1f}] {r['model']} · {r['word']}")
        out.append(f"      {r['sentence']}")
        out.append(f"      judge: {r['judge_note']}")
    return "\n".join(out)
