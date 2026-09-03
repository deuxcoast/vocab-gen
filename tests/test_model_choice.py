import pytest

from vocab_gen.generate import DEFAULT_MODEL, resolve_model


def test_defaults_to_opus(monkeypatch):
    monkeypatch.delenv("VOCAB_MODEL", raising=False)
    assert resolve_model() == DEFAULT_MODEL == "claude-opus-5"


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
