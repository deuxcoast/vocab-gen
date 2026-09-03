"""Exercise the HTTP layer end to end with the model call stubbed out."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from vocab_gen import server as server_mod
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


def stub_generate(result):
    """Match generate()'s signature in one place, so it drifts in one place."""

    def _stub(word, words, n=3, model=None, prefer=None, avoid=None, effort=None):
        return result, None, "test-model", "low"

    return _stub


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(
        History, "load", classmethod(lambda cls, path=None: History(tmp_path / "u.json"))
    )
    monkeypatch.setattr(server_mod, "extract_terms", lambda **kw: ["zephyr", "ziggurat"])
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
