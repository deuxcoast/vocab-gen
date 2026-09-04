import pytest

from vocab_gen.generate import DEFAULT_MODEL, resolve_model


def test_falls_back_to_the_configured_default(monkeypatch):
    """Asserts the invariant, not the literal — DEFAULT_MODEL is a tuning knob."""
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model() == DEFAULT_MODEL


def test_default_and_fallback_name_real_providers():
    from vocab_gen.generate import FALLBACK_MODEL, split_spec
    from vocab_gen.providers import PROVIDERS

    for spec in (DEFAULT_MODEL, FALLBACK_MODEL):
        provider, model = split_spec(spec)
        assert provider in PROVIDERS, spec
        assert model, spec


def test_default_and_fallback_are_different():
    from vocab_gen.generate import FALLBACK_MODEL

    assert DEFAULT_MODEL != FALLBACK_MODEL


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("opus", "claude-opus-5"),
        ("sonnet", "claude-sonnet-5"),
        ("haiku", "claude-haiku-4-5"),
        ("HAIKU", "claude-haiku-4-5"),
    ],
)
def test_aliases_expand(alias, expected, monkeypatch):
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model(alias) == expected


def test_full_id_passes_through(monkeypatch):
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_env_var_is_used_when_no_flag(monkeypatch):
    monkeypatch.setenv("VOCAB_MODEL", "haiku")
    assert resolve_model() == "claude-haiku-4-5"


def test_explicit_flag_beats_env_var(monkeypatch):
    monkeypatch.setenv("VOCAB_MODEL", "haiku")
    assert resolve_model("opus") == "claude-opus-5"


from vocab_gen.generate import (
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    resolve_effort,
    supports_effort,
)


def test_effort_defaults(monkeypatch):
    monkeypatch.delenv("VOCAB_EFFORT", raising=False)
    assert resolve_effort() == DEFAULT_EFFORT
    assert DEFAULT_EFFORT in EFFORT_LEVELS


def test_effort_env_var_and_flag_precedence(monkeypatch):
    monkeypatch.setenv("VOCAB_EFFORT", "max")
    assert resolve_effort() == "max"
    assert resolve_effort("low") == "low"


def test_effort_is_case_insensitive(monkeypatch):
    monkeypatch.delenv("VOCAB_EFFORT", raising=False)
    assert resolve_effort("HIGH") == "high"


def test_unknown_effort_is_rejected(monkeypatch):
    monkeypatch.delenv("VOCAB_EFFORT", raising=False)
    with pytest.raises(SystemExit) as exc:
        resolve_effort("turbo")
    assert "turbo" in str(exc.value)


@pytest.mark.parametrize(
    "model, ok",
    [
        ("claude-opus-5", True),
        ("claude-sonnet-5", True),
        # These 400 if effort is sent, so it must be dropped rather than passed.
        ("claude-haiku-4-5", False),
        ("claude-sonnet-4-5", False),
    ],
)
def test_effort_support_by_model(model, ok):
    assert supports_effort(model) is ok
