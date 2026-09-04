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
    # None | "openai_effort" | "qwen_thinking" | "thinking_toggle" | "anthropic_effort"
    reasoning: str | None = None
    # Explicit cache_control markers, vs. automatic prefix caching server-side.
    explicit_cache: bool = False
    default_model: str = ""
    notes: str = ""
    # True when the endpoint is account-specific and cannot be defaulted.
    needs_base_url: bool = False
    # Vendor and product names diverge (Zhipu/GLM, Alibaba/DashScope/Qwen), so
    # accept whichever the user reached for.
    key_aliases: tuple[str, ...] = ()


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
        # Official docs advertise "Json Output" but document only json_mode;
        # third-party sources claim strict schema on V4 Pro. Assume the weaker
        # of the two — the fallback works either way and reports which ran.
        json_schema=False,
        # Takes reasoning_effort but ignores it; the thinking switch is what works.
        reasoning="thinking_toggle",
        default_model="deepseek-v4-flash",
        notes=(
            "Automatic prefix caching, on by default. Peak pricing (01:00-04:00 "
            "and 06:00-10:00 UTC, Mon-Fri) is double off-peak."
        ),
    ),
    "moonshot": ProviderConfig(
        name="moonshot",
        kind="openai",
        base_url="https://api.moonshot.ai/v1",
        key_env="MOONSHOT_API_KEY",
        key_aliases=("KIMI_API_KEY",),
        json_schema=True,
        reasoning="thinking_toggle",
        default_model="kimi-k2.6",
        notes=(
            "Kimi. Docs moved to platform.kimi.ai; the API host is unverified — "
            "override VOCAB_MOONSHOT_BASE_URL if calls 404. The mainland host is "
            "api.moonshot.cn/v1. K3 ranks 2nd on EQ-Bench creative writing but "
            "costs more per card than Claude Sonnet 5."
        ),
    ),
    "zhipu": ProviderConfig(
        name="zhipu",
        kind="openai",
        base_url="https://api.z.ai/api/paas/v4",
        key_env="ZHIPUAI_API_KEY",
        key_aliases=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        # glm-4.7-flash accepts response_format=json_schema and then ignores it,
        # returning prose. Ask for json_object and inline the schema instead.
        json_schema=False,
        reasoning="thinking_toggle",
        default_model="glm-4.7-flash",
        notes=(
            "GLM. Mainland endpoint is open.bigmodel.cn/api/paas/v4. "
            "Defaults to the free glm-4.7-flash; the 5.x models need account "
            "balance. Note glm-4.7-flash reasons heavily (~300 tokens for one "
            "sentence), so give it output headroom. GLM-5.2 weights are MIT."
        ),
    ),
    "dashscope": ProviderConfig(
        name="dashscope",
        kind="openai",
        # Docs describe a workspace-scoped host, but the legacy shared endpoint
        # still answers for many accounts. Try it and let the error say otherwise.
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        key_env="DASHSCOPE_API_KEY",
        key_aliases=("QWEN_API_KEY", "ALIBABA_API_KEY"),
        json_schema=True,  # strict schema confirmed on 3.7/3.8 series
        reasoning="qwen_thinking",
        default_model="qwen3.8-flash",
        notes=(
            "Alibaba Qwen. Set VOCAB_DASHSCOPE_BASE_URL to "
            "https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1 "
            "(or cn-beijing). Strict json_schema on the 3.7/3.8 series. New "
            "accounts get a large free trial; Qwen3.5 weights are Apache-2.0."
        ),
    ),
    "minimax": ProviderConfig(
        name="minimax",
        kind="openai",
        base_url="https://api.minimax.chat/v1",
        key_env="MINIMAX_API_KEY",
        key_aliases=("MINIMAXI_API_KEY",),
        json_schema=False,  # unverified; fallback path is safe
        default_model="MiniMax-M2.7",
        notes="Open weights. Host unverified — override if calls fail.",
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


# Published rates in USD per million tokens, as (input, cached input, output).
# Checked 2026-09-04; vendors move these, so treat any cost the spike reports as
# indicative and re-check before a real decision. DeepSeek is quoted off-peak.
PRICING_CHECKED = "2026-09-04"
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "anthropic:claude-opus-5": (5.00, 0.50, 25.00),
    "anthropic:claude-sonnet-5": (2.00, 0.20, 10.00),
    "anthropic:claude-haiku-4-5": (1.00, 0.10, 5.00),
    "deepseek:deepseek-v4-flash": (0.22, 0.007, 0.66),
    "deepseek:deepseek-v4-pro": (0.66, 0.022, 1.98),
    "zhipu:glm-5.3": (1.40, 0.26, 4.40),
    "zhipu:glm-5.2": (1.40, 0.26, 4.40),
    "zhipu:glm-5.3-flash": (0.075, 0.015, 0.25),
    "zhipu:glm-4.7-flash": (0.0, 0.0, 0.0),
    "zhipu:glm-4.5-flash": (0.0, 0.0, 0.0),
    "moonshot:kimi-k3": (3.00, 0.30, 15.00),
    "moonshot:kimi-k2.6": (0.95, 0.16, 4.00),
    "moonshot:kimi-k2.5": (0.60, 0.10, 3.00),
    "moonshot:kimi-k2.7-code": (0.95, 0.19, 4.00),
    "dashscope:qwen3.8-max": (2.00, 0.25, 6.00),
    "dashscope:qwen3.8-flash": (0.10, 0.0125, 0.40),
    "dashscope:qwen3.5-flash": (0.10, 0.0125, 0.40),
    "minimax:MiniMax-M2.7": (0.30, 0.06, 1.20),
}


def cost_of(spec: str, usage) -> float | None:
    """Dollars for one call, or None when the model is not in the price table."""
    rates = MODEL_PRICING.get(spec)
    if rates is None:
        return None
    price_in, price_cached, price_out = rates
    return (
        usage.input_tokens * price_in
        + usage.cache_read * price_cached
        + usage.cache_write * price_in * 1.25  # writes carry a premium
        + usage.output_tokens * price_out
    ) / 1_000_000


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
    names = (f"VOCAB_{cfg.name.upper()}_API_KEY", cfg.key_env, *cfg.key_aliases)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def available(provider: str) -> bool:
    """Do we have what we need to call this provider at all?"""
    cfg = config_for(provider)
    if cfg.needs_base_url and not cfg.base_url:
        return False
    return bool(api_key_for(cfg))


def why_unavailable(provider: str) -> str:
    """A sentence explaining what is missing, for the CLI and the spike."""
    cfg = config_for(provider)
    if cfg.needs_base_url and not cfg.base_url:
        return f"needs VOCAB_{provider.upper()}_BASE_URL ({cfg.notes})"
    if not api_key_for(cfg):
        return f"no {cfg.key_env}"
    return ""


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
            user = _with_schema(user, schema)
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
        parsed, error = self._parse(text, schema_model)

        # Capability flags cannot be trusted: some vendors accept a json_schema
        # request and then ignore it. If schema mode produced nothing usable,
        # fall back to json_object with the schema inlined and try once more.
        retried = False
        if parsed is None and self.cfg.json_schema:
            retried = True
            structured = "json_object (after schema ignored)"
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": _with_schema(user, schema)},
                ],
                **{**kwargs, "response_format": {"type": "json_object"}},
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            parsed, error = self._parse(text, schema_model)

        if parsed is None and error is not None:
            raise ProviderError(error)

        u = getattr(response, "usage", None)
        cached = 0
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
        return Completion(
            parsed=parsed,
            text=text,
            usage=Usage(
                input_tokens=max((getattr(u, "prompt_tokens", 0) or 0) - cached, 0),
                output_tokens=getattr(u, "completion_tokens", 0) or 0,
                cache_read=cached,
            ),
            model=model,
            effort=effort,
            stop_reason=choice.finish_reason,
            structured=structured if retried else structured,
        )

    def _parse(self, text: str, schema_model):
        """(parsed, error) — error only when JSON was found but did not fit."""
        payload = extract_json(text)
        if payload is None:
            return None, None
        try:
            return schema_model.model_validate(payload), None
        except ValidationError as exc:
            return None, (
                f"{self.cfg.name}: returned JSON that does not match the schema "
                f"({exc.error_count()} errors)."
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
        if self.cfg.reasoning == "thinking_toggle":
            # GLM, DeepSeek and Kimi share this switch, and it is the only one
            # that works: DeepSeek ignores reasoning_effort (3303 output tokens
            # either way) and Kimi gets *worse* with it (7982 tokens, 100s).
            # Disabled, both drop to under 200 tokens and a few seconds.
            state = "enabled" if effort not in (None, "low") else "disabled"
            return {"extra_body": {"thinking": {"type": state}}}
        return {}


def _with_schema(user: str, schema: dict[str, Any]) -> str:
    """Without schema enforcement the shape has to travel in the prompt."""
    return (
        f"{user}\n\nReturn a single JSON object and nothing else, matching this "
        f"schema exactly — use these field names verbatim:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


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
