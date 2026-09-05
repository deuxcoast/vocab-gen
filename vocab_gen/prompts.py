"""Prompt variants, so the prompt itself can be measured rather than guessed at.

Two rules this module exists to enforce.

Variants are *composed*, not rewritten. Each is a recombination of the same
blocks, so an ablation differs from the baseline in exactly the thing it claims
to test. Hand-rewriting a whole prompt for each variant is how you end up
attributing a result to the rule you changed on purpose rather than the three
words you changed by accident.

Every variant states a hypothesis. If you cannot say in advance what a variant
should do to which metric, you are not running an experiment, you are fishing —
and with enough variants something always wins by chance.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- blocks ----------------------------------------------------------------

ROLE = """\
You write example sentences for a native English speaker's personal Anki vocabulary deck.

The learner's method: the front of a card is a single real sentence containing the target \
word; the back is one or two terse definition bullets. Sentences have historically been \
lifted from dictionary example banks and literary quotations, so they read like published \
prose — journalism, criticism, fiction, popular science — never like textbook filler.

Your job is to write fresh sentences for a new target word that ALSO happen to reuse words \
the learner is already studying, so that old vocabulary resurfaces in new contexts."""

WRITING = """\
How to write them:

- Write a sentence that is genuinely about something — a specific scene, claim, or event. \
Concrete beats abstract. A reader who did not know it was a vocabulary exercise should not \
be able to tell.
- 15 to 40 words. Vary the syntax across candidates; do not open every sentence the same way.
- Give each candidate a clearly different subject matter and register from the others.
- Write plain prose only. No markdown, no asterisks, no bold or italics, no HTML in the \
sentence — the card applies its own formatting.
- The target word must carry real semantic weight. Do NOT gloss or define it in the \
sentence — no appositives like "the nexus, or central link, between…". The card has to test \
recall, so context should suggest the meaning without handing it over.
- The target may be inflected to fit (plural, past tense, adverb form). Report the exact \
surface form you used."""

# The rule under test. The strict form is what ships today; the permissive form
# is the same policy with the "rather reuse nothing" clause relaxed.
REUSE_STRICT = """\
Reusing the learner's known words — this is the part that goes wrong most easily:

- Reuse ONE or TWO known words per sentence, and only where that word is genuinely the one \
a good writer would have reached for anyway.
- It is much better to reuse nothing than to force a pairing. A sentence that reads like two \
vocabulary words were bolted together has failed, even if both words are used correctly. If \
a candidate has no natural pairing, return an empty `reused` list and let the sentence stand \
on its own merit.
- Do not cluster words just because they are both "difficult". Ask whether the two words \
plausibly belong to the same subject matter, register, and era.
- Never reuse a known word that is a synonym or near-synonym of the target — it makes the \
sentence redundant and the card ambiguous.
- Only list a word in `reused` if it actually appears in that sentence."""

REUSE_PERMISSIVE = """\
Reusing the learner's known words:

- Reuse ONE or TWO known words per sentence. Reach for one wherever it can be made to fit \
without damaging the sentence — the whole point of the deck is that old words resurface, and \
a sentence that reuses nothing does no reinforcing work.
- Prefer a pairing that shares subject matter or register, but a slightly unexpected pairing \
is still worth having if the sentence reads well.
- Never reuse a known word that is a synonym or near-synonym of the target — it makes the \
sentence redundant and the card ambiguous.
- Only list a word in `reused` if it actually appears in that sentence."""

# Between the two. Keeps the domain-overlap bar that makes a reuse worth having,
# but removes the "rather reuse nothing" clause, which the measured run suggests
# is taken as an easy out: permissive reuse doubled the reuse rate, so the
# capability is there and the strict wording is what suppresses it.
REUSE_BALANCED = """\
Reusing the learner's known words:

- Reuse ONE or TWO known words per sentence. The test is whether the word shares the \
sentence's subject matter, register or situation — whether a good writer might have reached \
for it here, not merely whether it is used correctly.
- Search the list properly before concluding that nothing fits. Most subjects can be steered \
towards one of these words without strain, and across your candidates most should carry one. \
Choosing the sentence first and then looking is what makes a pairing hard to find; consider \
the word first and build the scene around it.
- If a word would have to be bent to fit — wrong register, wrong era, wrong subject — leave \
it out and let the sentence stand on its own. A forced pairing is worse than none.
- Never reuse a known word that is a synonym or near-synonym of the target — it makes the \
sentence redundant and the card ambiguous.
- Only list a word in `reused` if it actually appears in that sentence."""

DEFINITIONS = """\
The definition bullets: terse and plain, the way someone writes for their own recall. One \
short clause is often enough ("Unyielding."; "A process that can't be stopped."). Add a \
second bullet only when it earns its place — a distinct sense, or the connotation that makes \
the word worth knowing. Do not repeat the target word inside its own definition."""

TERSE = """\
You write example sentences for a personal vocabulary deck. Each sentence contains one \
target word and reads like published prose — journalism, criticism, fiction — not like a \
textbook exercise.

- 15 to 40 words, plain text, no markdown.
- Do not define the target in the sentence; context should suggest the meaning, not hand it \
over.
- Where one fits naturally, reuse a word from the learner's known-word list. Reusing nothing \
beats forcing a word in.
- Give one or two terse definition bullets."""


@dataclass(frozen=True)
class PromptVariant:
    name: str
    hypothesis: str
    blocks: tuple[str, ...]
    glosses: bool = True  # send the learner's definition with each preferred word
    avoid: bool = True  # send the recently-used words to skip
    kept: bool = True  # send previously kept sentences as calibration
    relatedness: bool = True  # ask for domain overlap when choosing a reuse
    # Generate this many times the requested candidates and show the best. Part
    # of the variant rather than a run-level flag so two arms differing only in
    # over-sampling can be judged side by side, in one run, on the same words.
    oversample: int = 1
    # Separate calls of n candidates each, pooled. Distinct from oversample:
    # asking one call for 2n makes the model spread its candidates across more
    # ground, since the prompt tells it to vary them, which appears to lower the
    # reuse rate of the pool being ranked. Two calls each see a normal request.
    batches: int = 1
    # How the preferred-word sample is biased: "fsrs", "lapses", or "uniform".
    weighting: str = "fsrs"

    def system_text(self) -> str:
        return "\n\n".join(self.blocks)


BASELINE = PromptVariant(
    name="baseline",
    hypothesis="What ships today. The reference every other variant is measured against.",
    blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
)

VARIANTS: dict[str, PromptVariant] = {
    "baseline": BASELINE,
    "permissive-reuse": PromptVariant(
        name="permissive-reuse",
        hypothesis=(
            "Relaxing 'rather reuse nothing' should raise the reuse rate. The open "
            "question is whether naturalness and reuse_fit fall with it — this is the "
            "72%-to-37% change from earlier, run as a controlled experiment."
        ),
        blocks=(ROLE, WRITING, REUSE_PERMISSIVE, DEFINITIONS),
        relatedness=False,
    ),
    "balanced-reuse": PromptVariant(
        name="balanced-reuse",
        hypothesis=(
            "Baseline reuses in only 33% of sentences; permissive reaches 65% but "
            "loses 0.70 naturalness. This keeps the domain-overlap bar, drops the "
            "'rather reuse nothing' escape, and tells the model to pick the word "
            "before building the scene. Predicts reuse above baseline with "
            "naturalness within noise of it."
        ),
        blocks=(ROLE, WRITING, REUSE_BALANCED, DEFINITIONS),
    ),
    "baseline-os2": PromptVariant(
        name="baseline-os2",
        hypothesis=(
            "The shipped prompt, generating six candidates and showing the best "
            "three. Predicts reuse well above baseline's 49% at best-of-three "
            "quality roughly flat, for 1.43x the cost — candidates share one "
            "call, so only the output scales."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        oversample=2,
    ),
    "baseline-os3": PromptVariant(
        name="baseline-os3",
        hypothesis=(
            "As above with nine candidates. Tests whether the gain from "
            "over-sampling keeps paying or flattens out."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        oversample=3,
    ),
    "baseline-b2": PromptVariant(
        name="baseline-b2",
        hypothesis=(
            "Two separate calls of three rather than one call of six. Tests the "
            "explanation offered for over-sampling's failure: if asking for six "
            "at once dilutes the pool, two normal requests should not, and "
            "ranking over them should raise reuse. If reuse stays flat here too, "
            "dilution was the wrong explanation."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        batches=2,
    ),
    "baseline-lapses": PromptVariant(
        name="baseline-lapses",
        hypothesis=(
            "The shipped prompt, choosing preferred words by the hand-rolled "
            "lapse score FSRS replaced. Predicts worse targeting — reused words "
            "less likely to have been forgotten — with sentence quality "
            "unchanged, since only which words are offered differs."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        weighting="lapses",
    ),
    "baseline-uniform": PromptVariant(
        name="baseline-uniform",
        hypothesis=(
            "The control: preferred words drawn with no weighting at all. If "
            "FSRS does not beat this, the weighting is decoration."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        weighting="uniform",
    ),
    "no-glosses": PromptVariant(
        name="no-glosses",
        hypothesis=(
            "Ablation: do the learner's definitions attached to preferred words earn "
            "their tokens? If reuse_fit and invented-reuse are unchanged without them, "
            "they do not."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        glosses=False,
    ),
    "no-avoid": PromptVariant(
        name="no-avoid",
        hypothesis=(
            "Ablation: does the recently-used list do anything the preferred list does "
            "not already do? Expect little change in quality; watch repetition."
        ),
        blocks=(ROLE, WRITING, REUSE_STRICT, DEFINITIONS),
        avoid=False,
    ),
    "terse": PromptVariant(
        name="terse",
        hypothesis=(
            "The instruction block is ~650 tokens. If a quarter of it performs as well, "
            "the rest is cargo — and cheaper prompts cache and iterate faster."
        ),
        blocks=(TERSE,),
    ),
}


def get(name: str | None) -> PromptVariant:
    if not name:
        return BASELINE
    try:
        return VARIANTS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown prompt variant {name!r}. Known: {', '.join(sorted(VARIANTS))}."
        ) from None
