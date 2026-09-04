"""The golden set.

Words deliberately absent from the deck, so a run reproduces the real
situation: a new word arriving with 795 known ones behind it. Chosen to spread
across part of speech and across the concrete/abstract axis, because that axis
is where the model's word-selection bias showed up — concrete nouns are easy to
build a scene around and abstract ones fight you.

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
)


def subset(n: int | None = None) -> tuple[Case, ...]:
    """First n cases — for a cheap smoke run without changing the set."""
    return GOLDEN if n is None else GOLDEN[:n]
