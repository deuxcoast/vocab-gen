"""Words held back from the prompt.

Some cards are worth keeping in a deck but not worth putting in front of a model
as a word to reuse. Slurs are the clear case: a generated sentence containing
one is not a card anyone wants, however correct the usage. So are words that are
not slurs but are routinely mistaken for them — "niggardly" is unrelated in
origin to any slur, and a sentence built around it invites exactly the
misreading the learner does not need.

This filters the known-word list sent to the model. It does **not** filter the
deck, and it does not stop you generating a card *for* one of these words: if
you ask for "niggardly" as a target you get it. The rule is about what the tool
volunteers, not about what you may study.

This list was written while a prompt variant was being rejected by Alibaba's
safety classifier, on the theory that a word list containing a slur was scoring
high enough to tip it over. That theory was tested afterwards and is wrong: with
the account on a paid tier, the same variant passes 10 of 10 attempts whether
the slurs are in the list or not. The exclusion stands on its own merit — a
generated sentence containing a slur is not a card anyone wants — and buys
nothing from any classifier.
"""

from __future__ import annotations

import os
from pathlib import Path

# Kept deliberately short. This is not an attempt to enumerate every slur — it
# covers what has turned up plus the obvious neighbours, and the user file below
# is the way to extend it.
DEFAULT_EXCLUSIONS = frozenset(
    {
        # Present in the deck when this was written.
        "maricón",
        "maricon",
        "jigaboo",
        "niggardly",
        # Not slurs, but read as them often enough to be unusable in an example.
        "niggling",
        "gypped",
        "welshed",
        # Ethnic and racial slurs.
        "coolie",  # in the deck; its own gloss calls it dated and offensive
        "coon",
        "wetback",
        "gook",
        "kike",
        "spic",
        "chink",
        "dago",
        "wop",
        "squaw",
        "redskin",
        "negress",
        "mulatto",
        "octoroon",
        "quadroon",
        "half-caste",
        # Slurs for sexuality and gender.
        "faggot",
        "tranny",
        "shemale",
        "catamite",
        # Slurs for disability.
        "retard",
        "retarded",
        "mongoloid",
        "spastic",
        "cretin",
    }
)


def user_file() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "vocab-gen" / "excluded.txt"


def load(path: Path | None = None) -> frozenset[str]:
    """Defaults plus anything in the user's list. Additive, never subtractive.

    One word per line; blank lines and lines starting with # are ignored.
    """
    path = Path(path) if path is not None else user_file()
    extra: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            word = line.split("#", 1)[0].strip().lower()
            if word:
                extra.add(word)
    except (FileNotFoundError, OSError):
        pass
    return frozenset(DEFAULT_EXCLUSIONS | extra)


def is_excluded(term: str, excluded: frozenset[str] | None = None) -> bool:
    excluded = load() if excluded is None else excluded
    return term.strip().lower() in excluded
