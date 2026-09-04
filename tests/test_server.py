"""Exercise the HTTP layer end to end with the model call stubbed out."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from vocab_gen import server as server_mod
from vocab_gen.collection import VocabWord
from vocab_gen.history import History


class FakeCandidate:
    def __init__(self, sentence, surface_form, reused):
        self.sentence, self.surface_form, self.reused = sentence, surface_form, reused


class FakeResult:
    part_of_speech = "noun"
    definition = ["A central link.", "The middle of it."]
    candidates = [
        # 'zephyr' is in the fake deck; 'bogus' is not and must be filtered out.
        FakeCandidate("The nexus of trade, carried on a zephyr.", "nexus", ["zephyr", "bogus"]),
    ]


class _Usage:
    input_tokens = output_tokens = cache_read = cache_write = 0
    cache_read_input_tokens = cache_creation_input_tokens = 0


def stub_generate(result):
    """Match generate()'s signature in one place, so it drifts in one place."""

    def _stub(
        word, words, n=3, model=None, prefer=None, avoid=None, effort=None,
        kept=None, allow_fallback=True,
    ):
        from vocab_gen.generate import Outcome

        return Outcome(result, _Usage(), "test-model", "low")

    return _stub


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Redirect history to a temp file, but keep real load/save semantics so a
    # save made by the handler is visible to a later load.
    real_load = History.load.__func__
    monkeypatch.setattr(
        History,
        "load",
        classmethod(lambda cls, path=None: real_load(cls, tmp_path / "u.json")),
    )
    monkeypatch.setattr(
        server_mod,
        "extract_vocab",
        lambda **kw: [VocabWord("zephyr"), VocabWord("ziggurat")],
    )
    monkeypatch.setattr("vocab_gen.generate.generate", stub_generate(FakeResult()))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod._handler("General", None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_serves_the_page(client):
    status, body = _get(client + "/")
    assert status == 200 and b"<title>vocab</title>" in body


def test_words_endpoint(client):
    status, body = _get(client + "/api/words")
    assert status == 200
    assert json.loads(body) == {"count": 2, "words": ["zephyr", "ziggurat"]}


def test_generate_returns_wrapped_front_and_filtered_reuse(client):
    status, data = _post(client + "/api/generate", {"word": "nexus", "n": 1})
    assert status == 200
    cand = data["candidates"][0]
    assert "<i><u>nexus</u></i>" in cand["front_html"]
    # 'bogus' is not in the deck, so the server must not credit it.
    assert cand["reused"] == ["zephyr"]


def test_reuse_not_present_in_sentence_is_not_credited(client, monkeypatch):
    """A word the model claims but did not actually write must not be shown."""
    class Claimed(FakeResult):
        candidates = [FakeCandidate("A plain sentence with nexus.", "nexus", ["ziggurat"])]

    monkeypatch.setattr("vocab_gen.generate.generate", stub_generate(Claimed()))
    _, data = _post(client + "/api/generate", {"word": "nexus", "n": 1})
    assert data["candidates"][0]["reused"] == []
    assert data["back_html"] == "<ul><li>A central link.</li><li>The middle of it.</li></ul>"


def test_empty_word_is_rejected(client):
    status, data = _post(client + "/api/generate", {"word": "  "})
    assert status == 400 and "error" in data


def test_unknown_route_404s(client):
    status, data = _post(client + "/api/nope", {})
    assert status == 404


def test_response_reports_the_model_used(client):
    _, data = _post(client + "/api/generate", {"word": "nexus", "n": 1})
    assert data["model"] == "test-model/low"


def test_choose_records_the_kept_candidate(client, tmp_path):
    """Copying a candidate in the UI is the signal that it was kept."""
    status, _ = _post(
        client + "/api/choose",
        {"word": "nexus", "sentence": "The nexus near the strait.", "reused": ["strait"]},
    )
    assert status == 200
    kept = History.load(tmp_path / "u.json").recent_kept()
    assert kept and kept[-1]["target"] == "nexus"
    assert kept[-1]["reused"] == ["strait"]


def test_giveaway_is_surfaced_to_the_page(client):
    _, data = _post(client + "/api/generate", {"word": "nexus", "n": 1})
    assert "giveaway" in data["candidates"][0]


def test_api_errors_reach_the_page_with_their_kind(client, monkeypatch):
    """A single-user tool should show what broke, not hide it."""
    from vocab_gen.generate import GenerationError

    def boom(*a, **kw):
        raise GenerationError("balance", "dashscope", "qwen3.8-flash", "Out of credit.")

    monkeypatch.setattr("vocab_gen.generate.generate", boom)
    status, data = _post(client + "/api/generate", {"word": "nexus", "n": 1})
    assert status == 502
    assert data["kind"] == "balance"
    assert data["provider"] == "dashscope"
    assert "Out of credit." in data["error"]


def test_a_fallback_is_reported_to_the_page(client, monkeypatch):
    from vocab_gen.generate import GenerationError, Outcome

    def fell_back(*a, **kw):
        out = Outcome(FakeResult(), _Usage(), "moonshot:kimi-k2.6", "low")
        out.fell_back_from = GenerationError(
            "rate_limit", "dashscope", "qwen3.8-flash", "Rate limited."
        )
        return out

    monkeypatch.setattr("vocab_gen.generate.generate", fell_back)
    status, data = _post(client + "/api/generate", {"word": "nexus", "n": 1})
    assert status == 200
    assert data["fell_back_from"]["provider"] == "dashscope"
    assert data["fell_back_from"]["kind"] == "rate_limit"


def test_send_creates_a_note(client, monkeypatch):
    from vocab_gen import anki

    calls = {}
    monkeypatch.setattr(anki, "is_duplicate", lambda f, b: False)
    monkeypatch.setattr(
        anki, "add_note",
        lambda f, b, allow_duplicate=False: calls.update(front=f, back=b) or 99,
    )
    status, data = _post(
        client + "/api/send",
        {"front": "<i><u>x</u></i>", "back": "<ul><li>d</li></ul>", "word": "x"},
    )
    assert status == 200 and data["note_id"] == 99
    assert calls["front"] == "<i><u>x</u></i>"


def test_send_reports_a_duplicate_rather_than_creating_one(client, monkeypatch):
    from vocab_gen import anki

    monkeypatch.setattr(anki, "is_duplicate", lambda f, b: True)
    monkeypatch.setattr(anki, "add_note", lambda *a, **kw: pytest.fail("must not add"))
    status, data = _post(client + "/api/send", {"front": "f", "back": "b", "word": "x"})
    assert status == 409 and data["duplicate"] is True


def test_send_anyway_overrides_the_duplicate_check(client, monkeypatch):
    from vocab_gen import anki

    seen = {}
    monkeypatch.setattr(anki, "is_duplicate", lambda f, b: True)
    monkeypatch.setattr(
        anki, "add_note",
        lambda f, b, allow_duplicate=False: seen.update(allow=allow_duplicate) or 7,
    )
    status, data = _post(
        client + "/api/send",
        {"front": "f", "back": "b", "word": "x", "allow_duplicate": True},
    )
    assert status == 200 and seen["allow"] is True


def test_send_records_the_card_as_kept(client, tmp_path, monkeypatch):
    from vocab_gen import anki

    monkeypatch.setattr(anki, "is_duplicate", lambda f, b: False)
    monkeypatch.setattr(anki, "add_note", lambda f, b, allow_duplicate=False: 1)
    _post(client + "/api/send",
          {"front": "f", "back": "b", "word": "nexus", "sentence": "s", "reused": ["strait"]})
    kept = History.load(tmp_path / "u.json").recent_kept()
    assert kept and kept[-1]["target"] == "nexus"


def test_send_with_anki_closed_explains_itself(client, monkeypatch):
    from vocab_gen import anki

    def closed(*a, **kw):
        raise anki.AnkiError("Could not reach AnkiConnect. Anki has to be running.")

    monkeypatch.setattr(anki, "is_duplicate", closed)
    status, data = _post(client + "/api/send", {"front": "f", "back": "b"})
    assert status == 502 and "running" in data["error"]


def test_send_rejects_an_empty_front(client):
    status, data = _post(client + "/api/send", {"front": "  ", "back": "b"})
    assert status == 400
