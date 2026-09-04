import json
import types

import pytest
from pydantic import BaseModel

from vocab_gen.generate import Generation, resolve_model, split_spec, supports_effort
from vocab_gen.providers import (
    MODEL_PRICING,
    OpenAICompatProvider,
    ProviderError,
    SystemBlock,
    Usage,
    _strictify,
    api_key_for,
    available,
    config_for,
    cost_of,
    extract_json,
    why_unavailable,
)


# --- model specs ------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
        ("deepseek:deepseek-chat", ("deepseek", "deepseek-chat")),
        ("ollama:qwen3:8b", ("ollama", "qwen3:8b")),  # model ids may contain colons
    ],
)
def test_split_spec(spec, expected):
    assert split_spec(spec) == expected


def test_bare_provider_name_resolves_to_its_default(monkeypatch):
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model("deepseek") == "deepseek:" + config_for("deepseek").default_model


def test_anthropic_aliases_still_work(monkeypatch):
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model("opus") == "claude-opus-5"


def test_effort_only_sent_where_supported():
    assert supports_effort("claude-opus-5") is True
    assert supports_effort("claude-haiku-4-5") is False
    # DeepSeek V4 does take reasoning_effort — an earlier version of this file
    # assumed it did not, which would have benchmarked it with reasoning off.
    assert supports_effort("deepseek:deepseek-v4-flash") is True
    assert supports_effort("dashscope:qwen3.8-flash") is True
    assert supports_effort("minimax:MiniMax-M2.7") is False


def test_unknown_provider_is_reported():
    with pytest.raises(ProviderError) as exc:
        config_for("nope")
    assert "Unknown provider" in str(exc.value)


def test_base_url_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VOCAB_DEEPSEEK_BASE_URL", "http://localhost:9999/v1")
    assert config_for("deepseek").base_url == "http://localhost:9999/v1"


def test_vocab_prefixed_key_wins_over_vendor_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "vendor")
    monkeypatch.setenv("VOCAB_DEEPSEEK_API_KEY", "override")
    assert api_key_for(config_for("deepseek")) == "override"


def test_local_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert available("ollama") is True


def test_missing_key_means_unavailable(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("VOCAB_MINIMAX_API_KEY", raising=False)
    assert available("minimax") is False


# --- JSON recovery ----------------------------------------------------------


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_markdown_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_buried_in_prose():
    text = 'Sure! Here is the object:\n{"a": 1}\nHope that helps.'
    assert extract_json(text) == {"a": 1}


def test_extract_json_gives_up_cleanly():
    assert extract_json("no json at all") is None
    assert extract_json("") is None


def test_extract_json_ignores_non_objects():
    assert extract_json("[1, 2, 3]") is None


# --- strict schema ----------------------------------------------------------


def test_strictify_adds_additional_properties_recursively():
    schema = {"type": "object", "properties": {"x": {"type": "object", "properties": {}}}}
    out = _strictify(schema)
    assert out["additionalProperties"] is False
    assert out["properties"]["x"]["additionalProperties"] is False


# --- OpenAI-compatible provider (mocked; no network) ------------------------


class Tiny(BaseModel):
    value: str


def _fake_openai(monkeypatch, content, *, captured=None, cached=0):
    """Install a stand-in openai.OpenAI that records the kwargs it is given."""

    class Completions:
        def create(self, **kwargs):
            if captured is not None:
                captured.update(kwargs)
            usage = types.SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                prompt_tokens_details=types.SimpleNamespace(cached_tokens=cached),
            )
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=content), finish_reason="stop"
            )
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class Client:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", Client)


def _complete(provider, **over):
    kwargs = dict(
        model="m",
        system=[SystemBlock("instructions"), SystemBlock("word list", cacheable=True)],
        user="go",
        schema_model=Tiny,
        max_tokens=100,
    )
    kwargs.update(over)
    return provider.complete(**kwargs)


def test_schema_capable_provider_sends_json_schema(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "hi"}', captured=captured)
    out = _complete(OpenAICompatProvider(config_for("dashscope")))
    assert captured["response_format"]["type"] == "json_schema"
    assert out.structured == "schema"
    assert out.parsed.value == "hi"


def test_json_object_provider_falls_back_and_inlines_the_schema(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '```json\n{"value": "hi"}\n```', captured=captured)
    out = _complete(OpenAICompatProvider(config_for("deepseek")))
    assert captured["response_format"] == {"type": "json_object"}
    # Without schema enforcement the schema has to travel in the prompt.
    user_msg = captured["messages"][1]["content"]
    assert "value" in user_msg and "schema" in user_msg.lower()
    assert out.structured == "json_object"
    assert out.parsed.value == "hi"  # fenced JSON still recovered


def test_system_blocks_are_merged_for_openai_dialect(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
    _complete(OpenAICompatProvider(config_for("deepseek")))
    system = captured["messages"][0]
    assert system["role"] == "system"
    assert "instructions" in system["content"] and "word list" in system["content"]


def test_cached_tokens_are_not_double_counted(monkeypatch):
    _fake_openai(monkeypatch, '{"value": "x"}', cached=80)
    out = _complete(OpenAICompatProvider(config_for("deepseek")))
    # prompt_tokens includes cached ones; input_tokens should be the fresh remainder.
    assert out.usage.cache_read == 80
    assert out.usage.input_tokens == 20


def test_schema_violation_is_reported_not_swallowed(monkeypatch):
    _fake_openai(monkeypatch, '{"wrong_field": 1}')
    with pytest.raises(ProviderError) as exc:
        _complete(OpenAICompatProvider(config_for("deepseek")))
    assert "does not match the schema" in str(exc.value)


def test_unparseable_output_yields_no_parse_rather_than_crashing(monkeypatch):
    _fake_openai(monkeypatch, "I'd rather write you a poem.")
    out = _complete(OpenAICompatProvider(config_for("deepseek")))
    assert out.parsed is None
    assert out.text.startswith("I'd rather")


def test_reasoning_effort_is_translated_per_vendor(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
    _complete(OpenAICompatProvider(config_for("openai")), effort="max")
    # Our five levels fold into the three these vendors accept.
    assert captured["reasoning_effort"] == "high"


def test_no_reasoning_param_for_providers_without_one(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
    _complete(OpenAICompatProvider(config_for("minimax")), effort="high")
    assert "reasoning_effort" not in captured


def test_usage_exposes_anthropic_field_names():
    u = Usage(input_tokens=1, output_tokens=2, cache_read=3, cache_write=4)
    assert u.cache_read_input_tokens == 3
    assert u.cache_creation_input_tokens == 4


def test_generation_schema_survives_strictify():
    """The real schema must still be valid after the strict-mode rewrite."""
    out = _strictify(Generation.model_json_schema())
    assert json.dumps(out)  # serialisable
    assert out["additionalProperties"] is False


# --- endpoints that cannot be defaulted --------------------------------------


def test_workspace_scoped_provider_is_unavailable_without_a_base_url(monkeypatch):
    """Qwen's endpoint is account-specific; a hardcoded guess would just 404."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.delenv("VOCAB_DASHSCOPE_BASE_URL", raising=False)
    assert available("dashscope") is False
    assert "BASE_URL" in why_unavailable("dashscope")


def test_workspace_scoped_provider_becomes_available_with_one(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("VOCAB_DASHSCOPE_BASE_URL", "https://w.example/compatible-mode/v1")
    assert available("dashscope") is True
    assert why_unavailable("dashscope") == ""


def test_why_unavailable_names_the_missing_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("VOCAB_MINIMAX_API_KEY", raising=False)
    assert why_unavailable("minimax") == "no MINIMAX_API_KEY"


# --- costing -----------------------------------------------------------------


def test_cost_uses_the_cached_rate_for_cached_tokens():
    warm = cost_of("anthropic:claude-sonnet-5", Usage(input_tokens=1000, cache_read=5000, output_tokens=300))
    cold = cost_of("anthropic:claude-sonnet-5", Usage(input_tokens=6000, output_tokens=300))
    assert warm < cold


def test_cost_of_a_free_model_is_zero():
    assert cost_of("zhipu:glm-4.7-flash", Usage(input_tokens=9999, output_tokens=9999)) == 0.0


def test_unknown_model_has_no_price():
    assert cost_of("ollama:qwen3", Usage(input_tokens=100)) is None


def test_every_priced_model_names_a_real_provider():
    from vocab_gen.providers import PROVIDERS

    for spec in MODEL_PRICING:
        provider = spec.split(":", 1)[0]
        assert provider in PROVIDERS, spec


def test_pricing_is_ordered_cached_cheaper_than_input():
    for spec, (price_in, cached, _out) in MODEL_PRICING.items():
        assert cached <= price_in, spec
