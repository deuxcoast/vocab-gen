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
    assert supports_effort("moonshot:kimi-k2.6") is True
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


def test_provider_needing_an_endpoint_is_unavailable_without_one(monkeypatch):
    """For accounts where the host is account-specific, a guess would just 404."""
    from vocab_gen.providers import PROVIDERS, ProviderConfig

    cfg = ProviderConfig(
        name="scoped", kind="openai", base_url=None, needs_base_url=True,
        key_env="SCOPED_API_KEY",
    )
    monkeypatch.setitem(PROVIDERS, "scoped", cfg)
    monkeypatch.setenv("SCOPED_API_KEY", "k")
    assert available("scoped") is False
    assert "BASE_URL" in why_unavailable("scoped")

    monkeypatch.setenv("VOCAB_SCOPED_BASE_URL", "https://w.example/v1")
    assert available("scoped") is True


def test_product_name_env_vars_are_accepted(monkeypatch):
    """The vendor is Zhipu but the product is GLM; users reach for the product."""
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("VOCAB_ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "k")
    assert available("zhipu") is True

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("VOCAB_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "k")
    assert api_key_for(config_for("dashscope")) == "k"


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


# --- error interpretation ----------------------------------------------------


def test_billing_failure_is_not_reported_as_rate_limiting():
    """GLM returns 429 for an empty balance; 'wait and retry' is wrong advice."""
    from vocab_gen.generate import _explain

    class Exc(Exception):
        status_code = 429

    exc = Exc("Error code: 429 - {'error': {'code': '1113', 'message': "
              "'Insufficient balance or no resource package. Please recharge.'}}")
    message = _explain("zhipu", "glm-5.3", exc)
    assert "balance" in message.lower()
    assert "rate limit" not in message.lower()


def test_402_is_reported_as_billing():
    from vocab_gen.generate import _explain

    class Exc(Exception):
        status_code = 402

    assert "balance" in _explain("deepseek", "deepseek-v4-flash", Exc("payment")).lower()


def test_genuine_rate_limiting_still_says_so():
    from vocab_gen.generate import _explain

    class Exc(Exception):
        status_code = 429

    msg = _explain("zhipu", "glm-4.7-flash", Exc("Too many requests, slow down"))
    assert "rate limited" in msg.lower()


def test_deepseek_and_kimi_use_the_thinking_toggle(monkeypatch):
    """reasoning_effort is a trap on both: ignored by DeepSeek, worse on Kimi."""
    for provider in ("deepseek", "moonshot"):
        captured = {}
        _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
        _complete(OpenAICompatProvider(config_for(provider)), effort="low")
        assert captured["extra_body"]["thinking"] == {"type": "disabled"}, provider
        assert "reasoning_effort" not in captured, provider


def test_zhipu_thinking_is_disabled_at_low_effort(monkeypatch):
    """Left on, GLM burns the whole budget reasoning and returns empty content."""
    captured = {}
    _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
    _complete(OpenAICompatProvider(config_for("zhipu")), effort="low")
    assert captured["extra_body"]["thinking"] == {"type": "disabled"}


def test_zhipu_thinking_is_enabled_at_higher_effort(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
    _complete(OpenAICompatProvider(config_for("zhipu")), effort="high")
    assert captured["extra_body"]["thinking"] == {"type": "enabled"}


def test_qwen_thinking_flag_follows_effort(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "x"}', captured=captured)
    _complete(OpenAICompatProvider(config_for("dashscope")), effort="low")
    assert captured["extra_body"]["enable_thinking"] is False


def test_falls_back_when_a_vendor_ignores_json_schema(monkeypatch):
    """GLM accepts response_format=json_schema and returns prose anyway."""
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            body = "Sentence 1\nSome prose, no JSON." if len(calls) == 1 else '{"value": "ok"}'
            usage = types.SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None
            )
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content=body), finish_reason="stop"
            )
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class Client:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=Completions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", Client)
    out = _complete(OpenAICompatProvider(config_for("dashscope")))

    assert len(calls) == 2, "should retry once"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "schema exactly" in calls[1]["messages"][1]["content"]
    assert out.parsed.value == "ok"
    assert "ignored" in out.structured


def test_no_retry_when_schema_mode_works(monkeypatch):
    captured = {}
    _fake_openai(monkeypatch, '{"value": "hi"}', captured=captured)
    out = _complete(OpenAICompatProvider(config_for("dashscope")))
    assert out.structured == "schema"


def test_no_retry_for_providers_already_using_json_object(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            usage = types.SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None
            )
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content="not json"), finish_reason="stop"
            )
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class Client:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=Completions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", Client)
    out = _complete(OpenAICompatProvider(config_for("zhipu")))
    assert len(calls) == 1  # already the weaker mode; nothing to fall back to
    assert out.parsed is None


# --- rate-limit resilience ---------------------------------------------------


def test_rate_limits_are_retried(monkeypatch):
    from vocab_gen.generate import _with_retries

    monkeypatch.setattr("vocab_gen.generate.time.sleep", lambda _s: None)
    attempts = []

    class Limited(Exception):
        status_code = 429

    def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise Limited("Too many requests")
        return "ok"

    assert _with_retries("moonshot", "m", call) == "ok"
    assert len(attempts) == 3


def test_billing_429_is_not_retried(monkeypatch):
    """Backing off against an empty wallet just wastes time."""
    from vocab_gen.generate import _with_retries

    monkeypatch.setattr("vocab_gen.generate.time.sleep", lambda _s: None)
    attempts = []

    class Broke(Exception):
        status_code = 429

    def call():
        attempts.append(1)
        raise Broke("Insufficient balance or no resource package. Please recharge.")

    with pytest.raises(Broke):
        _with_retries("zhipu", "glm", call)
    assert len(attempts) == 1


def test_retries_give_up_eventually(monkeypatch):
    from vocab_gen.generate import RATE_LIMIT_RETRIES, _with_retries

    monkeypatch.setattr("vocab_gen.generate.time.sleep", lambda _s: None)
    attempts = []

    class Limited(Exception):
        status_code = 429

    def call():
        attempts.append(1)
        raise Limited("Too many requests")

    with pytest.raises(Limited):
        _with_retries("moonshot", "m", call)
    assert len(attempts) == RATE_LIMIT_RETRIES
