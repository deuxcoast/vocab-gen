import os

from vocab_gen.env import find_env_file, load_env, parse_env


def test_parses_simple_assignment():
    assert parse_env("ANTHROPIC_API_KEY=sk-ant-abc") == {"ANTHROPIC_API_KEY": "sk-ant-abc"}


def test_ignores_comments_and_blank_lines():
    text = "# a comment\n\nFOO=bar\n   \n# another\n"
    assert parse_env(text) == {"FOO": "bar"}


def test_strips_export_prefix():
    assert parse_env("export FOO=bar") == {"FOO": "bar"}


def test_strips_matching_quotes():
    assert parse_env('FOO="bar"') == {"FOO": "bar"}
    assert parse_env("FOO='bar'") == {"FOO": "bar"}


def test_keeps_hash_inside_quoted_value():
    assert parse_env('FOO="bar # not a comment"') == {"FOO": "bar # not a comment"}


def test_strips_trailing_comment_on_unquoted_value():
    assert parse_env("FOO=bar # trailing") == {"FOO": "bar"}


def test_empty_placeholder_is_skipped():
    """The shipped .env has an empty key; it must not mask a real exported var."""
    assert parse_env("ANTHROPIC_API_KEY=") == {}


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    load_env(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_load_env_sets_when_absent(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_env(tmp_path)
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"


def test_finds_env_in_a_parent_directory(tmp_path):
    (tmp_path / ".env").write_text("FOO=bar")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_env_file(nested) == tmp_path / ".env"
