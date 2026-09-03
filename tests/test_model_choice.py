import pytest

from vocab_gen.generate import DEFAULT_MODEL, resolve_model


def test_falls_back_to_the_configured_default(monkeypatch):
    """Asserts the invariant, not the literal — DEFAULT_MODEL is a tuning knob."""
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model() == DEFAULT_MODEL


def test_default_is_a_known_model():
    from vocab_gen.generate import MODEL_ALIASES

    assert DEFAULT_MODEL in MODEL_ALIASES.values()


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
