"""Run the eval harness across models and print a comparison.

    uv run python scripts/eval.py                        # every ready provider
    uv run python scripts/eval.py anthropic dashscope    # specific ones
    uv run python scripts/eval.py --cases 5 --no-judge   # cheap smoke run
    uv run python scripts/eval.py --report <run_id>      # re-render a past run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vocab_gen.env import load_env  # noqa: E402
from vocab_gen.evals import report, runner  # noqa: E402
from vocab_gen.evals.judge import DEFAULT_JUDGE  # noqa: E402
from vocab_gen.providers import PROVIDERS, available  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="provider or provider:model specs")
    ap.add_argument(
        "--variants", nargs="*", default=None,
        help="prompt variants to compare (default: baseline only)",
    )
    ap.add_argument("--list-variants", action="store_true", help="show variants and exit")
    ap.add_argument("--cases", type=int, default=None, help="use only the first N cases")
    ap.add_argument("-n", "--candidates", type=int, default=3)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--no-judge", action="store_true", help="programmatic grading only")
    ap.add_argument("--label", default="", help="tag appended to the run id")
    ap.add_argument("--report", default=None, help="re-render a stored run and exit")
    args = ap.parse_args()

    load_env()

    if args.list_variants:
        from vocab_gen.prompts import VARIANTS

        for name, v in VARIANTS.items():
            print(f"\n{name}")
            print(f"  {v.hypothesis}")
        return 0

    if args.report:
        rows = runner.load(args.report)
        print(report.render(rows))
        print(report.render_paired(rows))
        print(report.examples(rows))
        return 0

    models = args.models or [p for p in PROVIDERS if available(p) and p != "ollama"]
    if not models:
        print("No providers ready. Set keys in .env.")
        return 1

    def event(kind, model, word, detail):
        mark = {"done": " ", "judged": "*", "error": "!"}.get(kind, " ")
        print(f"{mark} {model:30s} {word:16s} {detail}", file=sys.stderr)

    print(f"models: {', '.join(models)}", file=sys.stderr)
    run_id, rows = runner.run(
        models,
        variants=args.variants,
        n_cases=args.cases,
        n_candidates=args.candidates,
        oversample=args.oversample,
        judge_model=None if args.no_judge else args.judge,
        label=args.label,
        on_event=event,
    )
    print(f"\nrun {run_id} · {len(rows)} candidates\n")
    dicts = [r.__dict__ for r in rows]
    print(report.render(dicts))
    print(report.render_paired(dicts))
    print(report.examples(dicts))
    print(f"\nstored: evals/runs/{run_id}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
