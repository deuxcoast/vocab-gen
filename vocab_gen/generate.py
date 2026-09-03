"""Ask Claude for example sentences that reuse the user's existing vocabulary."""

from __future__ import annotations

import os

import anthropic
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-sonnet-5"

# Shorthands, so you can A/B with `--model haiku` instead of the full id.
MODEL_ALIASES = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}


def resolve_model(name: str | None = None) -> str:
    """Explicit argument beats $VOCAB_MODEL beats the default."""
    chosen = name or os.environ.get("VOCAB_MODEL") or DEFAULT_MODEL
    return MODEL_ALIASES.get(chosen.lower(), chosen)


class Candidate(BaseModel):
    sentence: str = Field(
        description="The example sentence, as plain text with no HTML."
    )
    surface_form: str = Field(
        description=(
            "The exact substring of `sentence` that is the target word or phrase, "
            "including any inflection actually used (e.g. 'chasms', 'adumbrated')."
        )
    )
    reused: list[str] = Field(
        description=(
            "Words from the learner's known-word list that appear in this sentence, "
            "spelled as they appear in the sentence. Empty list if none fit naturally."
        )
    )


class Generation(BaseModel):
    definition: list[str] = Field(
        description="One or two very short definition bullets for the target word."
    )
    part_of_speech: str = Field(
        description="e.g. 'adjective', 'noun', 'transitive verb'."
    )
    candidates: list[Candidate]


INSTRUCTIONS = """\
You write example sentences for a native English speaker's personal Anki vocabulary deck.

The learner's method: the front of a card is a single real sentence containing the target \
word; the back is one or two terse definition bullets. Sentences have historically been \
lifted from dictionary example banks and literary quotations, so they read like published \
prose — journalism, criticism, fiction, popular science — never like textbook filler.

Your job is to write fresh sentences for a new target word that ALSO happen to reuse words \
the learner is already studying, so that old vocabulary resurfaces in new contexts.

How to write them:

- Write a sentence that is genuinely about something — a specific scene, claim, or event. \
Concrete beats abstract. A reader who did not know it was a vocabulary exercise should not \
be able to tell.
- 15 to 40 words. Vary the syntax across candidates; do not open every sentence the same way.
- Give each candidate a clearly different subject matter and register from the others.
- The target word must carry real semantic weight. Do NOT gloss or define it in the \
sentence — no appositives like "the nexus, or central link, between…". The card has to test \
recall, so context should suggest the meaning without handing it over.
- The target may be inflected to fit (plural, past tense, adverb form). Report the exact \
surface form you used.

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
- Only list a word in `reused` if it actually appears in that sentence.

The definition bullets: terse and plain, the way someone writes for their own recall. One \
short clause is often enough ("Unyielding."; "A process that can't be stopped."). Add a \
second bullet only when it earns its place — a distinct sense, or the connotation that makes \
the word worth knowing. Do not repeat the target word inside its own definition.\
"""


def build_system(words: list[str]) -> list[dict]:
    """Static instructions + the known-word list, as a cacheable prefix.

    The word list is sorted deterministically upstream; an unstable order here
    would silently invalidate the cached prefix on every call.
    """
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {
            "type": "text",
            "text": (
                "The learner's known-word list follows. These are the words already in "
                f"the deck ({len(words)} of them):\n\n" + ", ".join(words)
            ),
            "cache_control": {"type": "ephemeral"},
        },
    ]


def generate(
    word: str,
    words: list[str],
    n: int = 3,
    model: str | None = None,
    prefer: list[str] | None = None,
    avoid: list[str] | None = None,
) -> tuple[Generation, object, str]:
    """Return the parsed generation, the raw usage object, and the model used."""
    model = resolve_model(model)
    if not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        raise SystemExit(
            "No Anthropic credentials found.\n"
            "Set one before generating:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "(Extraction still works without it: try `vocab --list-words`.)"
        )

    client = anthropic.Anthropic()
    try:
        response = _call(client, word, words, n, model, prefer, avoid)
    except anthropic.AuthenticationError:
        raise SystemExit(
            "Anthropic rejected the API key (401).\n"
            "Check the ANTHROPIC_API_KEY value in your .env, or in the shell if you "
            "exported one there — an exported variable overrides the file."
        ) from None
    except anthropic.NotFoundError:
        raise SystemExit(
            f"No such model: {model!r}.\n"
            "Try one of: opus, sonnet, haiku (or a full model id)."
        ) from None
    except anthropic.RateLimitError:
        raise SystemExit(
            "Rate limited by the Anthropic API. Wait a moment and retry."
        ) from None
    except anthropic.APIConnectionError:
        raise SystemExit(
            "Could not reach the Anthropic API. Check your connection."
        ) from None
    except anthropic.APIStatusError as exc:
        raise SystemExit(
            f"Anthropic API error {exc.status_code}: {exc.message}"
        ) from None

    parsed = response.parsed_output
    if parsed is None:
        raise SystemExit(
            f"Model returned no structured output (stop_reason={response.stop_reason})."
        )
    return parsed, response.usage, model


def build_user_message(
    word: str, n: int, prefer: list[str] | None = None, avoid: list[str] | None = None
) -> str:
    """The variable half of the prompt.

    Steering lives here rather than in the system prompt on purpose: this text
    sits after the last cache breakpoint, so it can change on every call without
    invalidating the cached word list.
    """
    parts = [
        f"Target word or phrase: {word}",
        "",
        f"Write {n} candidate sentences for it, following the method above. "
        "Also give the part of speech and the definition bullets.",
    ]
    if avoid:
        parts += [
            "",
            "These known words have come up in recent cards. Skip them unless one is "
            "the unmistakably right choice: " + ", ".join(avoid) + ".",
        ]
    if prefer:
        parts += [
            "",
            "These known words have rarely or never appeared. If any of them fits a "
            "sentence naturally, favour it — but the rule above still holds: reusing "
            "nothing beats forcing a word in. " + ", ".join(prefer) + ".",
        ]
    return "\n".join(parts)


def _call(
    client,
    word: str,
    words: list[str],
    n: int,
    model: str,
    prefer: list[str] | None = None,
    avoid: list[str] | None = None,
):
    return client.messages.parse(
        model=model,
        max_tokens=4000,
        system=build_system(words),
        output_format=Generation,
        messages=[{"role": "user", "content": build_user_message(word, n, prefer, avoid)}],
    )
