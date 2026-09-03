"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .collection import extract_terms
from .env import load_env
from .render import back_html, verified_reuse, wrap_target


def _style(enabled: bool):
    if not enabled:
        return lambda text, _code: text
    return lambda text, code: f"\033[{code}m{text}\033[0m"


def _print_result(word, result, words, model: str, color: bool) -> None:
    s = _style(color)
    known = {w.lower() for w in words}

    print()
    print(s(f"  {word}", "1;36"), s(f" · {result.part_of_speech} · {model}", "2"))
    print()

    for i, cand in enumerate(result.candidates, 1):
        verified = verified_reuse(cand.sentence, cand.reused, known)
        print(s(f"  {i}.", "1;33"), cand.sentence)
        front = wrap_target(cand.sentence, cand.surface_form)
        print(s(f"     {front}", "2"))
        if verified:
            print(s(f"     reuses: {', '.join(verified)}", "32"))
        else:
            print(s("     reuses: nothing (no natural fit)", "2"))
        print()

    print(s("  definition", "1;36"))
    for bullet in result.definition:
        print(f"    • {bullet}")
    print(s(f"    {back_html(result.definition)}", "2"))
    print()


def _print_html(result) -> None:
    for cand in result.candidates:
        print(wrap_target(cand.sentence, cand.surface_form))
    print()
    print(back_html(result.definition))


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
    parser.add_argument("--usage", action="store_true", help="report token usage after generating")
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
        )

    words = extract_terms(deck=args.deck, profile=args.profile)

    if args.list_words:
        for w in words:
            print(w)
        return 0

    if not args.word:
        parser.error("a word is required (or use --list-words / --serve)")

    if not words:
        print(f"No vocabulary words found in deck {args.deck!r}.", file=sys.stderr)
        return 1

    from .generate import generate

    result, usage, model = generate(args.word, words, n=args.count, model=args.model)

    if args.html:
        _print_html(result)
    else:
        _print_result(args.word, result, words, model, color=sys.stdout.isatty())

    if args.usage:
        print(
            f"  {model}: {usage.input_tokens} in / {usage.output_tokens} out"
            f" · cache write {usage.cache_creation_input_tokens}"
            f" · cache read {usage.cache_read_input_tokens}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
