"""The only module that writes to the collection, so it gets explicit tests."""

import json
import types

import pytest

from vocab_gen import anki


def fake_urlopen(monkeypatch, handler):
    """Capture the AnkiConnect request and return a canned reply."""
    sent = {}

    class Response:
        def __init__(self, body):
            self._body = json.dumps(body).encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(request, timeout=None):
        payload = json.loads(request.data)
        sent.update(payload)
        return Response(handler(payload))

    monkeypatch.setattr(anki.urllib.request, "urlopen", _open)
    return sent


def test_a_note_goes_to_the_right_deck_and_type(monkeypatch):
    sent = fake_urlopen(monkeypatch, lambda p: {"result": 1234, "error": None})
    assert anki.add_note("<i><u>obdurate</u></i>", "<ul><li>Unyielding.</li></ul>") == 1234
    note = sent["params"]["note"]
    assert note["deckName"] == "General"
    assert note["modelName"] == "code-article"
    assert set(note["fields"]) == {"Front", "Back"}


def test_notes_are_tagged_so_they_can_be_found_later(monkeypatch):
    sent = fake_urlopen(monkeypatch, lambda p: {"result": 1, "error": None})
    anki.add_note("front", "back")
    assert sent["params"]["note"]["tags"] == ["vocab-gen"]


def test_duplicates_are_refused_by_default(monkeypatch):
    sent = fake_urlopen(monkeypatch, lambda p: {"result": 1, "error": None})
    anki.add_note("front", "back")
    assert sent["params"]["note"]["options"]["allowDuplicate"] is False


def test_duplicates_can_be_allowed_explicitly(monkeypatch):
    sent = fake_urlopen(monkeypatch, lambda p: {"result": 1, "error": None})
    anki.add_note("front", "back", allow_duplicate=True)
    assert sent["params"]["note"]["options"]["allowDuplicate"] is True


def test_is_duplicate_reads_can_add_notes(monkeypatch):
    fake_urlopen(monkeypatch, lambda p: {"result": [False], "error": None})
    assert anki.is_duplicate("front", "back") is True
    fake_urlopen(monkeypatch, lambda p: {"result": [True], "error": None})
    assert anki.is_duplicate("front", "back") is False


def test_an_anki_error_is_surfaced_not_swallowed(monkeypatch):
    fake_urlopen(monkeypatch, lambda p: {"result": None, "error": "model was not found"})
    with pytest.raises(anki.AnkiError) as exc:
        anki.add_note("front", "back")
    assert "model was not found" in str(exc.value)


def test_a_closed_anki_says_to_open_anki(monkeypatch):
    def _open(request, timeout=None):
        raise anki.urllib.error.URLError("connection refused")

    monkeypatch.setattr(anki.urllib.request, "urlopen", _open)
    with pytest.raises(anki.AnkiError) as exc:
        anki.add_note("front", "back")
    assert "has to be running" in str(exc.value)


def test_available_is_false_when_anki_is_closed(monkeypatch):
    def _open(request, timeout=None):
        raise anki.urllib.error.URLError("nope")

    monkeypatch.setattr(anki.urllib.request, "urlopen", _open)
    assert anki.available() is False


def test_a_missing_note_id_is_an_error(monkeypatch):
    """Anki answering without creating anything must not look like success."""
    fake_urlopen(monkeypatch, lambda p: {"result": None, "error": None})
    with pytest.raises(anki.AnkiError):
        anki.add_note("front", "back")
