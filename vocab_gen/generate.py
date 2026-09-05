"""Ask a model for example sentences that reuse the user's existing vocabulary.

Which model is a runtime choice — see `providers.py` for the vendor differences
this module deliberately does not know about.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
import random
import time

from pydantic import BaseModel, Field, field_validator

from .prompts import BASELINE, PromptVariant, get as get_variant
from .providers import (
    PROVIDERS,
    classify,
    ProviderError,
    SystemBlock,
    available,
    build,
    config_for,
)

# Measured over 240 candidates on forty words. Kimi K2.6 with the balanced-reuse
# prompt reuses a deck word in 78% of sentences against Qwen's 40%, at judged
# quality that does not separate from Claude Sonnet 5, for $0.241 per 100
# accepted cards against Sonnet's $0.293. Reuse is the point of the deck, so it
# is worth the 22x over Qwen — which is 22x of a fifth of a cent.
#
# The prompt matters as much as the model here: the balanced variant costs Qwen
# 0.458 naturalness (real) and Kimi nothing (noise). Kimi follows the more
# demanding instruction; Qwen needs the escape clause. So the fallback keeps the
# baseline prompt, since it runs on Qwen.
DEFAULT_MODEL = "moonshot:kimi-k2.6"
DEFAULT_VARIANT = "balanced-reuse"
FALLBACK_MODEL = "dashscope:qwen3.8-flash"
FALLBACK_VARIANT = "baseline"

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
    @field_validator("definition")
    @classmethod
    def _drop_empty_bullets(cls, bullets: list[str]) -> list[str]:
        """Discard bullets carrying no words, and the leading-punctuation artifact.

        Models sometimes return a bullet that is punctuation only — ", " or
        ". " — or prefix a real one with a stray separator (": Dominance of one
        group over others."). Measured on a define-only pass over the 40-word
        golden set: 4 of 40 fully degenerate and 15 with a leading artifact on
        qwen3.8-flash, 2 and 4 on qwen3.8-max. It survives because nothing
        rejects it, and an empty definition then disables the giveaway check
        rather than failing loudly.

        Not raising: a good sentence with a bad definition is still worth
        showing. The definition simply becomes empty, and gives_away_answer
        reports "not assessable" instead of "clean".
        """
        cleaned = []
        for bullet in bullets:
            text = re.sub(r"^[^\w(]+", "", bullet).strip()
            if re.search(r"[^\W\d_]{2,}", text):
                cleaned.append(text)
        return cleaned
    part_of_speech: str = Field(
        description="e.g. 'adjective', 'noun', 'transitive verb'."
    )
    candidates: list[Candidate]


INSTRUCTIONS = BASELINE.system_text()  # kept for callers that referenced it


def build_system(words: list[str], variant: PromptVariant | None = None) -> list[SystemBlock]:
    """Static instructions + the known-word list, as a cacheable prefix.

    The word list is sorted deterministically upstream; an unstable order here
    would silently invalidate the cached prefix on every call.
    """
    variant = variant or BASELINE
    return [
        SystemBlock(variant.system_text()),
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
    variant: str | PromptVariant | None = None,
    oversample: int = 1,
    batches: int = 1,
) -> Outcome:
    """Generate candidates, falling back to a second model if the first cannot.

    The fallback is never silent: the failure that caused it is carried on the
    Outcome so the caller can show what went wrong. A card you can use plus a
    visible warning beats a bare error, and beats a card that quietly came from
    somewhere else.
    """
    spec = resolve_model(model)
    # Over-generating is cheap: the candidates share one call, so the cached
    # input is paid once and only the output scales. Six candidates cost 1.43x
    # three, not 2x.
    chosen = variant if isinstance(variant, PromptVariant) else get_variant(
        variant if variant is not None else DEFAULT_VARIANT
    )
    asked = max(1, n * max(1, oversample, chosen.oversample))
    rounds = max(1, batches, chosen.batches)
    try:
        return _pooled(
            rounds, spec, word, words, asked, prefer, avoid, effort, kept,
            variant if variant is not None else DEFAULT_VARIANT,
        )
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
            # The fallback runs on a different model, which needs a different
            # prompt: the balanced variant measurably degrades Qwen.
            outcome = _pooled(
                rounds, fallback, word, words, asked, prefer, avoid, effort, kept,
                variant if variant is not None else FALLBACK_VARIANT,
            )
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


def _pooled(rounds, spec, word, words, n, prefer, avoid, effort, kept, variant) -> Outcome:
    """One request, or several pooled into one Outcome.

    Deliberately sequential. Concurrent calls would halve the latency but would
    also race the prompt cache: the first request has not written it when the
    second starts, so both pay full price for the shared prefix.
    """
    first = _generate_once(spec, word, words, n, prefer, avoid, effort, kept, variant)
    if rounds <= 1:
        return first

    usage = first.usage
    for _ in range(rounds - 1):
        nxt = _generate_once(spec, word, words, n, prefer, avoid, effort, kept, variant)
        first.result.candidates.extend(nxt.result.candidates)
        usage = _add_usage(usage, nxt.usage)
    first.usage = usage
    return first


def _add_usage(a, b):
    from .providers import Usage

    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read=a.cache_read_input_tokens + b.cache_read_input_tokens,
        cache_write=a.cache_creation_input_tokens + b.cache_creation_input_tokens,
    )


def _generate_once(
    spec, word, words, n, prefer, avoid, effort, kept, variant=None
) -> Outcome:
    provider_name, model_id = split_spec(spec)
    variant = variant if isinstance(variant, PromptVariant) else get_variant(variant)
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
                system=build_system(words, variant),
                user=build_user_message(word, n, prefer, avoid, kept, variant),
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


RATE_LIMIT_RETRIES = 6
# A content filter is only worth a couple of attempts. Rate limiting clears with
# time, but a classifier scoring a fixed prompt does not change its mind: one
# variant here failed ~96% of full requests, and six retries each simply spent
# the account's quota on a failure that was never going to recover.
CONTENT_FILTER_RETRIES = 2


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


def _is_content_filter(exc: Exception) -> bool:
    """A provider-side safety classifier declining a benign prompt.

    Measured on Alibaba: one prompt variant passed 10 of 10 attempts and another
    passed 5 of 10, with nothing objectionable in either — the second simply sat
    nearer the classifier's threshold. Left unretried this silently biases any
    comparison to whichever requests happened through, exactly as rate limiting
    did on an earlier run.
    """
    body = str(exc).lower()
    return any(
        phrase in body
        for phrase in ("datainspectionfailed", "data_inspection_failed",
                       "inappropriate content", "content_filter")
    )


def _is_transient(exc: Exception) -> bool:
    return _is_rate_limit(exc) or _is_content_filter(exc)


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
            budget = (
                CONTENT_FILTER_RETRIES if _is_content_filter(exc) else RATE_LIMIT_RETRIES
            )
            if not _is_transient(exc) or attempt >= budget - 1:
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
    variant: PromptVariant | None = None,
) -> str:
    """The variable half of the prompt.

    Steering lives here rather than in the system prompt on purpose: this text
    sits after the last cache breakpoint, so it can change on every call without
    invalidating the cached word list.
    """
    variant = variant or BASELINE
    parts = [
        f"Target word or phrase: {word}",
        "",
        f"Write {n} candidate sentences for it, following the method above. "
        "Also give the part of speech and the definition bullets.",
    ]
    if avoid and variant.avoid:
        parts += [
            "",
            "These known words have come up in recent cards. Skip them unless one is "
            "the unmistakably right choice: " + ", ".join(avoid) + ".",
        ]
    if prefer:
        if variant.glosses:
            lines = []
            for w in prefer:
                gloss = (w.gloss or "").strip()
                if len(gloss) > 90:
                    gloss = gloss[:87].rstrip() + "..."
                lines.append(f"- {w.term}" + (f" — {gloss}" if gloss else ""))
            listing = "\n".join(lines)
            preamble = (
                "These known words have rarely or never appeared on a card, and are due "
                "to resurface. Each is given with the learner's own definition, which is "
                "the sense they actually learned — use that sense, not another one the "
                "word might carry."
            )
        else:
            listing = ", ".join(w.term for w in prefer)
            preamble = (
                "These known words have rarely or never appeared on a card, and are due "
                "to resurface."
            )
        parts += ["", preamble, "", listing]
        if variant.relatedness:
            parts += [
                "",
                "Pick from this list only where the word's subject matter genuinely "
                "overlaps with the sentence you are writing — a shared domain, register, "
                "or situation. Do not reach for one just because it is on the list: "
                "reusing nothing still beats forcing a word in.",
            ]
    if kept and variant.kept:
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


