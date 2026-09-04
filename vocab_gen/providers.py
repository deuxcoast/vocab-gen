"""Talking to more than one model vendor.

Every provider here answers the same question — "given a system prompt and a
user message, return an object matching this schema" — but they disagree about
almost all of the mechanics: how a system prompt is passed, whether prompt
caching is explicit or automatic, whether a JSON *schema* can be enforced or
only "some JSON", and how (or whether) reasoning depth is controlled.

This module is where those disagreements live, so the rest of the app can ask
for a `Generation` without caring who produces it.

The base URLs and model ids in PROVIDERS are a starting point, not gospel:
vendors move endpoints and rename models. Anything here can be overridden with
`VOCAB_<PROVIDER>_BASE_URL` / `VOCAB_<PROVIDER>_API_KEY`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


class ProviderError(RuntimeError):
    """A provider failed in a way worth reporting but not crashing over."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0

    # The CLI grew up against Anthropic's field names; keep them working.
    @property
    def cache_read_input_tokens(self) -> int:
        return self.cache_read

    @property
    def cache_creation_input_tokens(self) -> int:
        return self.cache_write


@dataclass(frozen=True)
class SystemBlock:
    """A chunk of system prompt, flagged for whether it is worth caching."""

    text: str
    cacheable: bool = False


@dataclass
class Completion:
    parsed: BaseModel | None
    text: str
    usage: Usage
    model: str
    effort: str | None = None
    stop_reason: str | None = None
    # How the JSON was obtained: schema-enforced, json-mode, or just asked for.
    structured: str = "schema"


# --------------------------------------------------------------------------
# Provider configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str  # "anthropic" | "openai"
    base_url: str | None = None
    key_env: str = ""
    # Whether response_format supports a real JSON *schema*, not just "some JSON".
    json_schema: bool = True
    # How reasoning depth is expressed, if at all.
    reasoning: str | None = None  # None | "openai_effort" | "qwen_thinking"
    # Explicit cache_control markers, vs. automatic prefix caching server-side.
    explicit_cache: bool = False
    default_model: str = ""
    notes: str = ""


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        kind="anthropic",
        key_env="ANTHROPIC_API_KEY",
        json_schema=True,
        reasoning="anthropic_effort",
        explicit_cache=True,
        default_model="claude-sonnet-5",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        kind="openai",
        base_url="https://api.deepseek.com",
        key_env="DEEPSEEK_API_KEY",
        json_schema=False,  # json_object mode; verify whether schema is supported
        reasoning=None,  # reasoning is a separate model, not a parameter
        default_model="deepseek-chat",
        notes="Automatic prefix caching server-side; reasoning via a separate model id.",
    ),
    "moonshot": ProviderConfig(
        name="moonshot",
        kind="openai",
        base_url="https://api.moonshot.cn/v1",
        key_env="MOONSHOT_API_KEY",
        json_schema=False,
        default_model="moonshot-v1-32k",
        notes="Kimi. International endpoint differs; override with VOCAB_MOONSHOT_BASE_URL.",
    ),
    "zhipu": ProviderConfig(
        name="zhipu",
        kind="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        key_env="ZHIPUAI_API_KEY",
        json_schema=False,
        default_model="glm-4-plus",
        notes="GLM family.",
    ),
    "dashscope": ProviderConfig(
        name="dashscope",
        kind="openai",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        key_env="DASHSCOPE_API_KEY",
        json_schema=True,
        reasoning="qwen_thinking",
        default_model="qwen-max",
        notes="Alibaba Qwen. Mainland endpoint drops the -intl.",
    ),
    "minimax": ProviderConfig(
        name="minimax",
        kind="openai",
        base_url="https://api.minimax.chat/v1",
        key_env="MINIMAX_API_KEY",
        json_schema=False,
        default_model="abab6.5s-chat",
    ),
    "ollama": ProviderConfig(
        name="ollama",
        kind="openai",
        base_url="http://localhost:11434/v1",
        key_env="",  # local, no key
        json_schema=True,
        default_model="qwen3",
        notes="Local open-weight models; zero marginal cost.",
    ),
    "openai": ProviderConfig(
        name="openai",
        kind="openai",
        base_url=None,  # SDK default
        key_env="OPENAI_API_KEY",
        json_schema=True,
        reasoning="openai_effort",
        default_model="gpt-4.1",
    ),
}


def config_for(provider: str) -> ProviderConfig:
    try:
        cfg = PROVIDERS[provider]
    except KeyError:
        raise ProviderError(
            f"Unknown provider {provider!r}. Known: {', '.join(sorted(PROVIDERS))}."
        ) from None
    override = os.environ.get(f"VOCAB_{provider.upper()}_BASE_URL")
    return ProviderConfig(**{**cfg.__dict__, "base_url": override}) if override else cfg


def api_key_for(cfg: ProviderConfig) -> str | None:
    if not cfg.key_env:
        return "not-needed"
    return os.environ.get(f"VOCAB_{cfg.name.upper()}_API_KEY") or os.environ.get(cfg.key_env)


def available(provider: str) -> bool:
    """Do we have what we need to call this provider at all?"""
    cfg = config_for(provider)
    return bool(api_key_for(cfg))


# --------------------------------------------------------------------------
# JSON recovery, for providers that cannot enforce a schema
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON out of a chat completion.

    Models without schema enforcement wrap JSON in prose, fence it as markdown,
    or emit it bare. Try each, cheapest first.
    """
    if not text:
        return None
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _candidates(text: str):
    yield text.strip()
    for match in _FENCE.findall(text):
        yield match.strip()
    # Outermost brace pair, for JSON buried in commentary.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        yield text[start : end + 1]


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class AnthropicProvider:
    kind = "anthropic"

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def complete(
        self,
        *,
        model: str,
        system: list[SystemBlock],
        user: str,
        schema_model: type[BaseModel],
        max_tokens: int,
        effort: str | None = None,
    ) -> Completion:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key_for(self.cfg))
        blocks: list[dict[str, Any]] = []
        for block in system:
            entry: dict[str, Any] = {"type": "text", "text": block.text}
            if block.cacheable:
                entry["cache_control"] = {"type": "ephemeral"}
            blocks.append(entry)

        kwargs: dict[str, Any] = {}
        if effort:
            kwargs["output_config"] = {"effort": effort}

        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=blocks,
            output_format=schema_model,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        u = response.usage
        return Completion(
            parsed=response.parsed_output,
            text="".join(b.text for b in response.content if b.type == "text"),
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read=u.cache_read_input_tokens or 0,
                cache_write=u.cache_creation_input_tokens or 0,
            ),
            model=model,
            effort=effort,
            stop_reason=response.stop_reason,
            structured="schema",
        )


class OpenAICompatProvider:
    """Anything speaking the OpenAI chat-completions dialect.

    Most Chinese vendors expose one of these, which is why a single class covers
    DeepSeek, Moonshot, Zhipu, Qwen, MiniMax and a local Ollama.
    """

    kind = "openai"

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def complete(
        self,
        *,
        model: str,
        system: list[SystemBlock],
        user: str,
        schema_model: type[BaseModel],
        max_tokens: int,
        effort: str | None = None,
    ) -> Completion:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key_for(self.cfg) or "not-needed",
            base_url=self.cfg.base_url,
        )

        # No explicit cache markers: these vendors cache prefixes automatically,
        # so the win comes from keeping the system text byte-stable, which it is.
        system_text = "\n\n".join(b.text for b in system)
        schema = schema_model.model_json_schema()

        kwargs: dict[str, Any] = {}
        structured = "schema"
        if self.cfg.json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__.lower(),
                    "schema": _strictify(schema),
                    "strict": True,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
            structured = "json_object"
            user = (
                f"{user}\n\nReturn a single JSON object and nothing else, matching "
                f"this schema exactly:\n{json.dumps(schema, ensure_ascii=False)}"
            )
        kwargs.update(self._reasoning_kwargs(effort))

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        choice = response.choices[0]
        text = choice.message.content or ""

        parsed = None
        payload = extract_json(text)
        if payload is not None:
            try:
                parsed = schema_model.model_validate(payload)
            except ValidationError as exc:
                raise ProviderError(
                    f"{self.cfg.name}: returned JSON that does not match the schema "
                    f"({exc.error_count()} errors)."
                ) from None

        u = getattr(response, "usage", None)
        cached = 0
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
        return Completion(
            parsed=parsed,
            text=text,
            usage=Usage(
                input_tokens=(getattr(u, "prompt_tokens", 0) or 0) - cached,
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                cache_read=cached,
            ),
            model=model,
            effort=effort,
            stop_reason=choice.finish_reason,
            structured=structured,
        )

    def _reasoning_kwargs(self, effort: str | None) -> dict[str, Any]:
        if not effort or not self.cfg.reasoning:
            return {}
        if self.cfg.reasoning == "openai_effort":
            # These vendors only know three levels; fold the top of ours together.
            level = {"xhigh": "high", "max": "high"}.get(effort, effort)
            return {"reasoning_effort": level}
        if self.cfg.reasoning == "qwen_thinking":
            return {"extra_body": {"enable_thinking": effort not in (None, "low")}}
        return {}


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Strict json_schema mode requires additionalProperties:false throughout."""
    if isinstance(schema, dict):
        out = {k: _strictify(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [_strictify(v) for v in schema]
    return schema


def build(provider: str):
    cfg = config_for(provider)
    return AnthropicProvider(cfg) if cfg.kind == "anthropic" else OpenAICompatProvider(cfg)
