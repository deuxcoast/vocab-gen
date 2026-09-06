"""The golden set.

Words chosen to sit outside the deck, so a run reproduces the real situation: a
new word arriving with the rest of the deck behind it. Spread across part of
speech and across the concrete/abstract axis, because that axis is where the
model's word-selection bias showed up — concrete nouns are easy to build a scene
around and abstract ones fight you.

Four of them are not in fact outside the deck, which was found while building
the embedding index rather than by reading this list: `verisimilitude` is in it
literally, and `descry`, `flagon` and `prevaricate` are in it by inflection, as
`descried`, `flagons` and `prevaricating`. That is a flaw in the set, and the
tempting fix — swapping those four out — is the wrong one, because changing the
list invalidates comparison against every earlier run. It is left standing and
handled instead: `History.plan` holds the target out of its own preferred list,
so the model is never offered the word it is being asked to teach, and
`detect_reuse` already excluded the target from reuse counts. What remains is
that for those four the deck is effectively one word smaller than for the other
thirty-six, which is small enough to live with and too small to see.

Keep this list stable. Changing it invalidates comparison against earlier runs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    word: str
    pos: str
    register: str  # "concrete" | "abstract"


GOLDEN: tuple[Case, ...] = (
    Case("hidebound", "adjective", "abstract"),
    Case("peremptory", "adjective", "abstract"),
    Case("risible", "adjective", "abstract"),
    Case("apostasy", "noun", "abstract"),
    Case("hegemony", "noun", "abstract"),
    Case("ennui", "noun", "abstract"),
    Case("verisimilitude", "noun", "abstract"),
    Case("tumult", "noun", "abstract"),
    Case("pariah", "noun", "concrete"),
    Case("quisling", "noun", "concrete"),
    Case("palanquin", "noun", "concrete"),
    Case("excoriate", "verb", "abstract"),
    Case("vitiate", "verb", "abstract"),
    Case("inveigh", "verb", "abstract"),
    Case("obviate", "verb", "abstract"),
    Case("bowdlerize", "verb", "abstract"),
    Case("maunder", "verb", "concrete"),
    Case("descry", "verb", "concrete"),
    Case("de facto", "phrase", "abstract"),
    Case("tour de force", "phrase", "abstract"),
    # Appended, never reordered: the first twenty remain a prefix, so subset(20)
    # still reproduces the set earlier runs used. Weighted towards concrete
    # nouns, which the original twenty were short of at 15 abstract to 5.
    Case("reliquary", "noun", "concrete"),
    Case("portcullis", "noun", "concrete"),
    Case("bandolier", "noun", "concrete"),
    Case("ossuary", "noun", "concrete"),
    Case("samovar", "noun", "concrete"),
    Case("trebuchet", "noun", "concrete"),
    Case("buttress", "noun", "concrete"),
    Case("cairn", "noun", "concrete"),
    Case("culvert", "noun", "concrete"),
    Case("flagon", "noun", "concrete"),
    Case("obloquy", "noun", "abstract"),
    Case("asperity", "noun", "abstract"),
    Case("rectitude", "noun", "abstract"),
    Case("venality", "noun", "abstract"),
    Case("arrogate", "verb", "abstract"),
    Case("prevaricate", "verb", "abstract"),
    Case("traduce", "verb", "abstract"),
    Case("winnow", "verb", "concrete"),
    Case("lambent", "adjective", "concrete"),
    Case("feckless", "adjective", "abstract"),
)


def subset(n: int | None = None) -> tuple[Case, ...]:
    """First n cases — for a cheap smoke run without changing the set."""
    return GOLDEN if n is None else GOLDEN[:n]
