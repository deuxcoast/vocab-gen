"""Ask a model for example sentences that reuse the user's existing vocabulary.

Which model is a runtime choice — see `providers.py` for the vendor differences
this module deliberately does not know about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import random
import time

from pydantic import BaseModel, Field

from .providers import (
    PROVIDERS,
    classify,
    ProviderError,
    SystemBlock,
    available,
    build,
    config_for,
)

# Measured over 234 candidates: Qwen3.8-Flash is statistically indistinguishable
# from Claude Sonnet 5 on judged quality and costs $0.011 per 100 accepted cards
# against Sonnet's $0.293. Kimi K2.6 also did not separate from Sonnet and has
# the best reuse rate, so it is the fallback when the primary cannot answer.
DEFAULT_MODEL = "dashscope:qwen3.8-flash"
FALLBACK_MODEL = "moonshot:kimi-k2.6"

# Shorthands, so you can A/B with `--model haiku` instead of the full id.
MODEL_ALIASES = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}


# Effort trades thinking depth against token spend. Measured on this prompt:
# Opus 5 went 866 -> 294 output tokens from high to low, a 2.4x cost cut, with no
# quality loss I could detect in the sentences.
DEFAULT_EFFORT = "low"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Models that reject output_config.effort outright (400). Sending it anyway
# would fail the whole request, so it is dropped for these.
EFFORT_UNSUPPORTED = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-haiku-3")

# Thinking tokens count against max_tokens, so a ceiling sized for low effort
# starves higher levels: the model spends the budget reasoning and never emits
# the structured output, returning stop_reason=max_tokens. The ceiling is only a
# cap — unused headroom costs nothing — but it stays at 16k, the safe limit for
# non-streaming requests.
# Only a ceiling — unused headroom costs nothing — but it has to clear the
# reasoning some providers do regardless of effort. GLM with thinking left on
# spent 4000 tokens reasoning and emitted no content at all.
MAX_TOKENS_BY_EFFORT = {
    None: 8000,
    "low": 8000,
    "medium": 12000,
    "high": 16000,
    "xhigh": 16000,
    "max": 16000,
}


def split_spec(spec: str) -> tuple[str, str]:
    """"deepseek:deepseek-chat" -> ("deepseek", "deepseek-chat").

    A bare model id means Anthropic, so every existing invocation still works.
    """
    provider, sep, model = spec.partition(":")
    return (provider, model) if sep else ("anthropic", spec)


def supports_effort(spec: str) -> bool:
    """Whether it is safe to send a reasoning-depth setting for this target."""
    provider, model = split_spec(spec)
    cfg = config_for(provider)
    if cfg.reasoning is None:
        return False
    if cfg.kind == "anthropic":
        return not any(model.startswith(prefix) for prefix in EFFORT_UNSUPPORTED)
    return True


def resolve_effort(name: str | None = None) -> str:
    """Explicit argument beats $VOCAB_EFFORT beats the default."""
    chosen = (name or os.environ.get("VOCAB_EFFORT") or DEFAULT_EFFORT).lower()
    if chosen not in EFFORT_LEVELS:
        raise SystemExit(
            f"Unknown effort {chosen!r}. Choose one of: {', '.join(EFFORT_LEVELS)}."
        )
    return chosen


def resolve_model(name: str | None = None) -> str:
    """Explicit argument beats $VOCAB_MODEL beats the default."""
    chosen = name or os.environ.get("VOCAB_MODEL") or DEFAULT_MODEL
    low = chosen.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    if low in PROVIDERS:  # bare provider name -> that provider's default model
        return f"{low}:{config_for(low).default_model}"
    return chosen


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
- Write plain prose only. No markdown, no asterisks, no bold or italics, no HTML \
in the sentence — the card applies its own formatting.
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


def build_system(words: list[str]) -> list[SystemBlock]:
    """Static instructions + the known-word list, as a cacheable prefix.

    The word list is sorted deterministically upstream; an unstable order here
    would silently invalidate the cached prefix on every call.
    """
    return [
        SystemBlock(INSTRUCTIONS),
        SystemBlock(
            "The learner's known-word list follows. These are the words already in "
            f"the deck ({len(words)} of them):\n\n" + ", ".join(words),
            cacheable=True,
        ),
    ]


class GenerationError(Exception):
    """A generation failed, with enough structure for a UI to show it usefully."""

    def __init__(
        self, kind: str, provider: str, model: str, message: str, also: str = ""
    ):
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.model = model
        self.message = message
        # The other provider that failed first, when a fallback was attempted.
        self.also = also

    @property
    def title(self) -> str:
        return f"{self.also} + {self.provider}" if self.also else self.provider

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
            "also": self.also,
            "title": self.title,
        }


@dataclass
class Outcome:
    """What a generation produced, plus how it got there."""

    result: "Generation"
    usage: object
    model: str
    effort: str | None
    # Set when the primary model could not answer and the fallback was used.
    fell_back_from: "GenerationError | None" = None


def generate(
    word: str,
    words: list[str],
    n: int = 3,
    model: str | None = None,
    prefer=None,
    avoid: list[str] | None = None,
    effort: str | None = None,
    kept: list[dict] | None = None,
    allow_fallback: bool = True,
) -> Outcome:
    """Generate candidates, falling back to a second model if the first cannot.

    The fallback is never silent: the failure that caused it is carried on the
    Outcome so the caller can show what went wrong. A card you can use plus a
    visible warning beats a bare error, and beats a card that quietly came from
    somewhere else.
    """
    spec = resolve_model(model)
    try:
        return _generate_once(spec, word, words, n, prefer, avoid, effort, kept)
    except GenerationError as primary:
        fallback = resolve_model(FALLBACK_MODEL)
        chose_explicitly = model is not None
        if (
            not allow_fallback
            or chose_explicitly  # an explicit --model is an instruction, not a hint
            or spec == fallback
            or primary.kind == "not_found"  # a typo will not be fixed by another vendor
            or not available(split_spec(fallback)[0])
        ):
            raise
        try:
            outcome = _generate_once(fallback, word, words, n, prefer, avoid, effort, kept)
        except GenerationError as secondary:
            # Both failed. Reporting only the second would point at the wrong
            # provider — the fallback was never the one you asked for.
            raise GenerationError(
                secondary.kind,
                secondary.provider,
                secondary.model,
                f"{primary.provider} failed, and so did the fallback.\n\n"
                f"{primary.provider} ({primary.kind}): {primary.message}\n\n"
                f"{secondary.provider} ({secondary.kind}): {secondary.message}",
                also=primary.provider,
            ) from None
        outcome.fell_back_from = primary
        return outcome


def _generate_once(
    spec, word, words, n, prefer, avoid, effort, kept
) -> Outcome:
    provider_name, model_id = split_spec(spec)
    cfg = config_for(provider_name)
    effort = resolve_effort(effort) if supports_effort(spec) else None

    if not available(provider_name):
        raise GenerationError(
            "auth", provider_name, model_id,
            f"No credentials for {provider_name}.\n"
            f"Set {cfg.key_env} (or VOCAB_{provider_name.upper()}_API_KEY) in your .env.",
        )

    provider = build(provider_name)
    try:
        completion = _with_retries(
            provider_name,
            model_id,
            lambda: provider.complete(
                model=model_id,
                system=build_system(words),
                user=build_user_message(word, n, prefer, avoid, kept),
                schema_model=Generation,
                max_tokens=MAX_TOKENS_BY_EFFORT.get(effort, 4000),
                effort=effort,
            ),
        )
    except ProviderError as exc:
        raise GenerationError("other", provider_name, model_id, str(exc)) from None
    except Exception as exc:  # every vendor SDK raises its own hierarchy
        raise GenerationError(
            classify(exc), provider_name, model_id, _explain(provider_name, model_id, exc)
        ) from None

    if completion.parsed is None:
        if completion.stop_reason in ("max_tokens", "length"):
            raise GenerationError(
                "other", provider_name, model_id,
                f"Ran out of output budget at effort={effort!r} before the model "
                "finished. Retry at a lower effort.",
            )
        raise GenerationError(
            "other", provider_name, model_id,
            f"{provider_name} returned no usable structured output "
            f"(stop_reason={completion.stop_reason}, mode={completion.structured}).",
        )
    return Outcome(completion.parsed, completion.usage, spec, effort)


RATE_LIMIT_RETRIES = 5


def _is_rate_limit(exc: Exception) -> bool:
    """A 429 that is about throughput, not an empty wallet."""
    if getattr(exc, "status_code", None) != 429 and "RateLimit" not in type(exc).__name__:
        return False
    body = str(exc).lower()
    return not any(
        phrase in body
        for phrase in ("insufficient balance", "no resource package", "recharge",
                       "arrears", "billing")
    )


def _with_retries(provider: str, model: str, call):
    """Back off through rate limiting rather than dropping the request.

    Vendor SDKs retry a couple of times by default, which is not enough on
    accounts with low throughput limits — an eval run lost 14 of 20 cases for
    one provider this way, and silently biased its sample to whichever cases
    happened to fall outside the limit window.
    """
    delay = 2.0
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return call()
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
    raise RuntimeError("unreachable")




def _explain(provider: str, model: str, exc: Exception) -> str:
    """Turn a vendor exception into something worth reading."""
    if isinstance(exc, ImportError):
        return (
            f"A Python package needed for {provider} is missing: {exc}.\n"
            "If you installed the CLI with `uv tool install`, reinstall it after a "
            "dependency change:\n"
            "  uv tool install --editable --force ~/deuxcoast/vocab-gen"
        )

    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    cfg = config_for(provider)
    if status == 401 or "Authentication" in name:
        return (
            f"{provider} rejected the API key (401).\n"
            f"Check {cfg.key_env} in your .env, or in the shell if you exported one "
            "there — an exported variable overrides the file."
        )
    if status == 404 or "NotFound" in name:
        return (
            f"{provider} has no model {model!r}.\n"
            f"Its default is {cfg.default_model!r}; endpoints and model ids move, so "
            f"check the vendor docs or override VOCAB_{provider.upper()}_BASE_URL."
        )
    # A 402, or a 429 whose body mentions money, is a billing problem. Reporting
    # it as rate limiting sends you off to wait instead of to top up.
    body = str(exc).lower()
    if status == 402 or any(
        phrase in body
        for phrase in ("insufficient balance", "no resource package", "recharge",
                       "quota", "arrears", "billing")
    ):
        return (
            f"{provider} reports no usable balance for {model!r}.\n"
            "Top up, or pick a model on its free tier."
        )
    if status == 429 or "RateLimit" in name:
        return f"Rate limited by {provider}. Wait a moment and retry."
    if "Connection" in name:
        return (
            f"Could not reach {provider} at {cfg.base_url or 'its default endpoint'}.\n"
            "Check your connection, or the base URL if you overrode it."
        )
    return f"{provider} error ({name}): {str(exc)[:300]}"


def build_user_message(
    word: str,
    n: int,
    prefer=None,
    avoid: list[str] | None = None,
    kept: list[dict] | None = None,
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
        lines = []
        for w in prefer:
            gloss = (w.gloss or "").strip()
            if len(gloss) > 90:
                gloss = gloss[:87].rstrip() + "..."
            lines.append(f"- {w.term}" + (f" — {gloss}" if gloss else ""))
        parts += [
            "",
            "These known words have rarely or never appeared on a card, and are due "
            "to resurface. Each is given with the learner's own definition, which is "
            "the sense they actually learned — use that sense, not another one the "
            "word might carry.",
            "",
            "\n".join(lines),
            "",
            "Pick from this list only where the word's subject matter genuinely "
            "overlaps with the sentence you are writing — a shared domain, register, "
            "or situation. Do not reach for one just because it is on the list: "
            "reusing nothing still beats forcing a word in.",
        ]
    if kept:
        examples = "\n".join(f"- {k['sentence']}" for k in kept if k.get("sentence"))
        if examples:
            parts += [
                "",
                "For calibration, here are sentences this learner chose to keep from "
                "earlier batches. Match their register and density; do not reuse their "
                "subject matter.",
                "",
                examples,
            ]
    return "\n".join(parts)
