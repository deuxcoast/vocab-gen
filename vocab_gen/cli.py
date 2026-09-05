"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .collection import extract_vocab, held_back
from .env import load_env
from .history import History
from .lexicon import (
    MIN_MARKS,
    Marks,
    calibration_sample,
    deck_report,
    frontier,
    suggest,
)
from .providers import PROVIDERS, available, why_unavailable
from .render import back_html, prepare, rank_candidates


def _style(enabled: bool):
    if not enabled:
        return lambda text, _code: text
    return lambda text, code: f"\033[{code}m{text}\033[0m"


def _print_result(word, result, cands, model, effort, color: bool) -> None:
    s = _style(color)
    tag = f"{model}/{effort}" if effort else model

    print()
    print(s(f"  {word}", "1;36"), s(f" · {result.part_of_speech} · {tag}", "2"))
    print()

    for i, c in enumerate(cands, 1):
        print(s(f"  {i}.", "1;33"), c["sentence"])
        print(s(f"     {c['front_html']}", "2"))
        if c["reused"]:
            note = ", ".join(c["reused"])
            if c.get("unclaimed"):
                note += f"  (unreported by the model: {', '.join(c['unclaimed'])})"
            print(s(f"     reuses: {note}", "32"))
        else:
            print(s("     reuses: nothing (no natural fit)", "2"))
        if c["missing_target"]:
            print(s(f"     ✗ does not use {word!r} at all — unusable", "1;31"))
        if c.get("wrong_sense"):
            print(s(f"     ⚠ used as {c['wrong_sense']}, not the sense on the card", "31"))
        if c["giveaway"]:
            print(s(f"     ⚠ gives the answer away: {', '.join(c['giveaway'])}", "31"))
        print()

    print(s("  definition", "1;36"))
    for bullet in result.definition:
        print(f"    • {bullet}")
    print(s(f"    {back_html(result.definition)}", "2"))
    print()


KIND_LABEL = {
    "auth": "API key rejected",
    "balance": "out of credit",
    "rate_limit": "rate limited",
    "not_found": "no such model",
    "connection": "cannot reach the API",
    "setup": "install is broken",
    "other": "failed",
}


def _print_failure(exc, color: bool) -> None:
    s = _style(color)
    label = KIND_LABEL.get(exc.kind, exc.kind)
    print(file=sys.stderr)
    print(s(f"  ✗ {exc.title} — {label}", "1;31"), file=sys.stderr)
    for line in exc.message.splitlines():
        print(f"    {line}", file=sys.stderr)
    print(file=sys.stderr)


def _print_fallback(exc, used: str, color: bool) -> None:
    s = _style(color)
    label = KIND_LABEL.get(exc.kind, exc.kind)
    print(file=sys.stderr)
    print(s(f"  ! {exc.provider} — {label}; used {used} instead", "1;33"), file=sys.stderr)
    for line in exc.message.splitlines():
        print(s(f"    {line}", "2"), file=sys.stderr)
    print(file=sys.stderr)


def check_providers(color: bool) -> int:
    """One tiny real call per configured provider, so failures are visible."""
    from .generate import GenerationError, _generate_once, resolve_model, split_spec

    s = _style(color)
    print()
    worst = 0
    for name in PROVIDERS:
        if not available(name):
            print(f"  {'-':2s} {name:12s} {why_unavailable(name)}")
            continue
        spec = resolve_model(name)
        try:
            _generate_once(spec, "obdurate", ["zephyr", "strait"], 1, None, None, None, None)
            print(s(f"  ok {name:12s} {spec}", "32"))
        except GenerationError as exc:
            label = KIND_LABEL.get(exc.kind, exc.kind)
            print(s(f"  ✗  {name:12s} {label}", "1;31"))
            print(s(f"     {exc.message.splitlines()[0]}", "2"))
            worst = 1
        except Exception as exc:  # never let one provider stop the sweep
            print(s(f"  ✗  {name:12s} {type(exc).__name__}: {str(exc)[:70]}", "1;31"))
            worst = 1
    print()
    return worst


def _ask_which_kept(cands: list[dict]) -> int | None:
    """Which candidate did you actually use? Skipped when not interactive."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        reply = input(f"  which did you keep? [1-{len(cands)}, enter to skip] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not reply.isdigit():
        return None
    idx = int(reply) - 1
    return idx if 0 <= idx < len(cands) else None


def run_calibration(vocab, n: int) -> int:
    """Ask which words are already known, and fit the frontier to the answers.

    Interactive because there is no way to infer it: population prevalence puts
    a learner in the right neighbourhood and cannot say where *they* stop. The
    sample deliberately spans both sides of any plausible frontier.
    """
    if not sys.stdin.isatty():
        print("--calibrate needs a terminal; use --known WORD... instead", file=sys.stderr)
        return 2
    marks = Marks.load()
    words = calibration_sample(vocab, n=n)
    print(f"  {len(words)} words, hardest last. y = I know it, n = I don't, s = skip, q = stop.\n")
    asked = 0
    for i, word in enumerate(words, 1):
        try:
            answer = input(f"  [{i}/{len(words)}] {word}  ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer.startswith("q"):
            break
        if answer.startswith("s") or not answer:
            continue
        marks.mark([word], known=answer.startswith("y"))
        asked += 1
    marks.save()
    band = frontier(marks, vocab)
    print(f"\n  {asked} answered · {len(marks)} marks in total")
    print(f"  frontier: {band.ceiling:+.2f} ({band.source})")
    if band.n_marks < MIN_MARKS:
        print(f"  still below {MIN_MARKS} marks, so the deck-derived band is being used")
    else:
        print("  words easier than this are treated as already known")
    print("\n  run  vocab --suggest  to see what it proposes now")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vocab",
        description="Generate Anki vocabulary sentences that reuse words you already study.",
    )
    parser.add_argument("word", nargs="?", help="the new word or phrase, e.g. 'sui generis'")
    parser.add_argument("-n", "--count", type=int, default=3, help="candidates (default 3)")
    parser.add_argument("--html", action="store_true", help="print paste-ready HTML only")
    parser.add_argument("--list-words", action="store_true", help="dump known words and exit")
    parser.add_argument("--deck", default="General", help="deck to read (default: General)")
    parser.add_argument("--profile", default=None, help="path to the Anki profile directory")
    parser.add_argument("--serve", action="store_true", help="run the local web interface")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    parser.add_argument("--open", action="store_true", help="open a browser with --serve")
    parser.add_argument(
        "--model",
        default=None,
        help="model to use: opus, sonnet, haiku, or a full id (env: VOCAB_MODEL)",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="thinking depth; lower is cheaper (default: low, env: VOCAB_EFFORT). "
        "Ignored by models that don't support it.",
    )
    parser.add_argument(
        "--oversample", type=int, default=1, metavar="N",
        help="generate N times as many candidates and show the best (default 1). "
        "Cheap: candidates share one call, so only the output scales.",
    )
    parser.add_argument(
        "--suggest", nargs="?", type=int, const=30, default=None, metavar="N",
        help="words worth learning next that are not in your deck (default 30)",
    )
    parser.add_argument(
        "--calibrate", nargs="?", type=int, const=40, default=None, metavar="N",
        help="mark N words known/unknown to locate your own difficulty frontier",
    )
    parser.add_argument(
        "--known", nargs="+", default=None, metavar="WORD",
        help="record words you already know, so they stop being suggested",
    )
    parser.add_argument("--usage", action="store_true", help="report token usage after generating")
    parser.add_argument(
        "--stats", action="store_true", help="show which deck words have been used, then exit"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="probe every configured provider and report its status, then exit",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="don't steer toward unused words, and don't record this run",
    )
    args = parser.parse_args(argv)

    load_env()  # a real exported var still wins over .env

    if args.serve:
        from .server import serve

        return serve(
            port=args.port,
            deck=args.deck,
            profile=args.profile,
            open_browser=args.open,
            model=args.model,
            effort=args.effort,
            oversample=args.oversample,
        )

    vocab = extract_vocab(deck=args.deck, profile=args.profile)
    words = [w.term for w in vocab]

    if args.list_words:
        for w in words:
            print(w)
        withheld = held_back(deck=args.deck, profile=args.profile)
        if withheld:
            print(
                f"\n({len(withheld)} held back from prompts: {', '.join(withheld)})",
                file=sys.stderr,
            )
        return 0

    if args.check:
        return check_providers(color=sys.stdout.isatty())

    if args.stats:
        cov = History.load().coverage(words)
        pct = 100 * cov["seen"] / cov["deck"] if cov["deck"] else 0
        print(f"  {cov['seen']} of {cov['deck']} deck words have appeared ({pct:.1f}%)")
        print(f"  {cov['unseen']} never used · {cov['uses']} reuses recorded")
        withheld = held_back(deck=args.deck, profile=args.profile)
        if withheld:
            print(f"  {len(withheld)} held back from prompts: {', '.join(withheld)}")
        with_state = sum(1 for w in vocab if w.has_memory_state)
        print(
            f"  {with_state} of {len(vocab)} carry FSRS memory state"
        )
        print("\n  most likely to have been forgotten (FSRS retrievability):")
        for w in sorted(vocab, key=lambda w: -w.shakiness)[:6]:
            r = w.retrievability()
            recall = f"{r:.0%}" if r is not None else " -- "
            print(
                f"    {w.term:20s} recall {recall}  stability {w.stability:6.1f}d  "
                f"lapses {w.lapses}"
            )
        if cov["top"]:
            print("\n  most used:")
            for w, n in cov["top"]:
                print(f"    {n:3d}  {w}")
        return 0

    if args.known is not None:
        marks = Marks.load()
        marks.mark(args.known, known=True)
        marks.save()
        print(f"  recorded {len(args.known)} known · {len(marks.known)} total")
        return 0

    if args.calibrate is not None:
        return run_calibration(vocab, args.calibrate)

    if args.suggest is not None:
        marks = Marks.load()
        report = deck_report(vocab, marks)
        band = report["band"]
        print(f"  {report['on_list']} of {report['deck']} deck words are on a curated list")
        print(f"  band {band.floor:+.2f} to {band.ceiling:+.2f} — {band.source}"
              + (f", {band.n_marks} marks" if band.n_marks else ""))
        if band.n_marks < MIN_MARKS:
            print("  this band is inferred from your deck, not measured; "
                  "run --calibrate to fit it to you")
        print(f"  {report['remaining']} candidates in band\n")
        for c in suggest(vocab, marks, n=args.suggest, band=band):
            print(f"    {c.word:<18} {c.n_lists:>2} lists   prevalence {c.prevalence:+.2f}")
        return 0

    if not args.word:
        parser.error("a word is required (or use --list-words / --serve)")

    if not words:
        print(f"No vocabulary words found in deck {args.deck!r}.", file=sys.stderr)
        return 1

    from .generate import generate

    history = None if args.no_history else History.load()
    prefer, avoid = history.plan(vocab) if history else ([], [])
    kept = history.recent_kept() if history else []

    from .generate import GenerationError

    try:
        outcome = generate(
            args.word,
            words,
            n=args.count,
            model=args.model,
            prefer=prefer,
            avoid=avoid,
            effort=args.effort,
            kept=kept,
            oversample=args.oversample,
        )
    except GenerationError as exc:
        _print_failure(exc, color=sys.stderr.isatty())
        return 1

    result, usage, model, effort = (
        outcome.result, outcome.usage, outcome.model, outcome.effort
    )
    if outcome.fell_back_from is not None:
        _print_fallback(outcome.fell_back_from, model, color=sys.stderr.isatty())
    cands = rank_candidates(prepare(result, args.word, words), args.count)

    if args.html:
        for c in cands:
            print(c["front_html"])
        print()
        print(back_html(result.definition))
    else:
        _print_result(args.word, result, cands, model, effort, color=sys.stdout.isatty())

    if history is not None:
        for c in cands:
            history.record(c["reused"])
        if not args.html:
            chosen = _ask_which_kept(cands)
            if chosen is not None:
                history.record_kept(
                    args.word, cands[chosen]["sentence"], cands[chosen]["reused"]
                )
        history.save()

    if args.usage:
        print(
            f"  {model}{'/' + effort if effort else ''}: "
            f"{usage.input_tokens} in / {usage.output_tokens} out"
            f" · cache write {usage.cache_creation_input_tokens}"
            f" · cache read {usage.cache_read_input_tokens}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
