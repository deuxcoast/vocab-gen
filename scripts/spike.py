"""Feasibility probe: can each provider actually do this job at all?

Not a quality benchmark — that is what the eval harness will be for. This
answers the cheaper questions first, so a model that cannot clear them never
reaches the expensive comparison:

  * can we reach it with the credentials we have?
  * does it return JSON matching our schema, and by which mechanism?
  * does it hallucinate deck membership? (our known cross-vendor failure)
  * what does one call cost in tokens, and does its prefix cache engage?

Usage:
    uv run python scripts/spike.py                # every provider with credentials
    uv run python scripts/spike.py deepseek zhipu # only these
    uv run python scripts/spike.py --word obdurate --repeat 2
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from vocab_gen.collection import extract_vocab  # noqa: E402
from vocab_gen.env import load_env  # noqa: E402
from vocab_gen.generate import generate, resolve_model, split_spec  # noqa: E402
from vocab_gen.history import History  # noqa: E402
from vocab_gen.providers import (  # noqa: E402
    PRICING_CHECKED,
    PROVIDERS,
    available,
    cost_of,
    why_unavailable,
)
from vocab_gen.render import prepare, unknown_claims  # noqa: E402


def probe(spec: str, word: str, vocab, words, prefer, avoid) -> dict:
    started = time.perf_counter()
    row = {"spec": spec, "ok": False, "note": "", "secs": 0.0}
    try:
        result, usage, used, effort = generate(
            word, words, n=3, model=spec, prefer=prefer, avoid=avoid
        )
    except SystemExit as exc:
        row["note"] = str(exc).splitlines()[0]
        row["secs"] = time.perf_counter() - started
        return row

    cands = prepare(result, word, words)
    claimed = sum(len(c.reused) for c in result.candidates)
    verified = sum(len(c["reused"]) for c in cands)
    invented = sum(len(unknown_claims(c.reused, words)) for c in result.candidates)
    row.update(
        ok=True,
        secs=time.perf_counter() - started,
        effort=effort,
        input=usage.input_tokens,
        output=usage.output_tokens,
        cache_read=usage.cache_read_input_tokens,
        cache_write=usage.cache_creation_input_tokens,
        claimed=claimed,
        verified=verified,
        invented=invented,
        giveaway=sum(1 for c in cands if c["giveaway"]),
        missing=sum(1 for c in cands if c["missing_target"]),
        sample=cands[0]["sentence"] if cands else "",
        cost=cost_of(spec, usage),
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("providers", nargs="*", help="provider or provider:model specs")
    ap.add_argument("--word", default="obdurate")
    ap.add_argument("--repeat", type=int, default=1, help="calls each (2+ shows caching)")
    args = ap.parse_args()

    load_env()
    vocab = extract_vocab()
    words = [w.term for w in vocab]
    prefer, avoid = History.load().plan(vocab)
    print(f"deck: {len(words)} words · target: {args.word}\n")

    specs = args.providers or [p for p in PROVIDERS if available(p)]
    if not specs:
        print("No providers have credentials. Set keys in .env, e.g. DEEPSEEK_API_KEY.")
        return 1

    skipped = []
    for name in PROVIDERS:
        if name not in specs and not available(name):
            skipped.append(f"{name} ({why_unavailable(name)})")

    rows = []
    for spec in specs:
        resolved = resolve_model(spec)
        provider, _ = split_spec(resolved)
        if not available(provider):
            rows.append({"spec": resolved, "ok": False, "note": "no credentials"})
            continue
        for _ in range(args.repeat):
            rows.append(probe(resolved, args.word, vocab, words, prefer, avoid))

    head = f"{'spec':30s} {'ok':3s} {'in':>6s} {'out':>6s} {'cw':>6s} {'cr':>6s} "
    head += f"{'claim':>5s} {'ver':>4s} {'inv':>4s} {'give':>4s} {'miss':>4s} {'secs':>5s} {'$/100':>7s}"
    print(head)
    print("-" * len(head))
    for r in rows:
        if not r.get("ok"):
            print(f"{r['spec']:30s} {'--':3s}  {r.get('note', '')[:66]}")
            continue
        cost = r.get("cost")
        money = f"{cost * 100:7.3f}" if cost is not None else "      ?"
        print(
            f"{r['spec']:30s} {'yes':3s} {r['input']:6d} {r['output']:6d} "
            f"{r['cache_write']:6d} {r['cache_read']:6d} {r['claimed']:5d} "
            f"{r['verified']:4d} {r['invented']:4d} {r['giveaway']:4d} "
            f"{r['missing']:4d} {r['secs']:5.1f} {money}"
        )

    print(
        "\nlegend: cw/cr=cache write/read · claim=reuses asserted · ver=verified "
        "real · inv=hallucinated · give=leaks the definition · miss=target word absent"
    )
    print(
        f"$/100 = projected dollars per 100 cards at this call's token mix; "
        f"rates checked {PRICING_CHECKED}, '?' means unpriced."
    )
    for r in rows:
        if r.get("ok") and r.get("sample"):
            print(f"\n--- {r['spec']}\n    {r['sample']}")
    if skipped:
        print("\nnot probed: " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
