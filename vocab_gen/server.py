"""A small local web interface for composing cards.

Read-only: it reads the Anki collection and never writes to it. Bound to
127.0.0.1 so it is not reachable from the network.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collection import extract_terms
from .history import History
from .render import back_html, verified_reuse, wrap_target

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vocab</title>
<style>
  :root {
    --bg: #fbfaf7; --fg: #1c1a17; --muted: #6b6560; --line: #e3ded6;
    --card: #ffffff; --accent: #7a4a1e; --ok: #2f6b43;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #171614; --fg: #ececea; --muted: #9a938c; --line: #2f2c28;
      --card: #1f1e1b; --accent: #d9a06a; --ok: #7fbf95;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.6 Charter, "Iowan Old Style", Georgia, serif;
  }
  .wrap { max-width: 46rem; margin: 0 auto; padding: 3rem 1.25rem 5rem; }
  h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 .2rem; letter-spacing: -.01em; }
  .sub { color: var(--muted); font-size: .85rem; margin-bottom: 2rem; }
  form { display: flex; gap: .5rem; margin-bottom: .75rem; }
  input[type=text] {
    flex: 1; padding: .7rem .85rem; font: inherit; color: var(--fg);
    background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  }
  input[type=text]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  select, button {
    font: inherit; padding: .7rem .9rem; border-radius: 8px;
    border: 1px solid var(--line); background: var(--card); color: var(--fg);
    cursor: pointer;
  }
  button.go { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button:disabled { opacity: .55; cursor: default; }
  .status { color: var(--muted); font-size: .85rem; min-height: 1.4rem; margin-bottom: 1.5rem; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 1.1rem 1.2rem; margin-bottom: .9rem;
  }
  .num { color: var(--muted); font-size: .8rem; font-family: ui-monospace, monospace; }
  .sentence { margin: .35rem 0 .8rem; }
  .sentence u { text-underline-offset: 3px; }
  .reuse { font-size: .82rem; color: var(--ok); font-family: ui-monospace, monospace; }
  .reuse.none { color: var(--muted); }
  .row { display: flex; gap: .4rem; margin-top: .8rem; flex-wrap: wrap; }
  .row button { padding: .35rem .6rem; font-size: .8rem; border-radius: 6px; }
  .defs { margin: 0; padding-left: 1.1rem; }
  .defs li { margin-bottom: .2rem; }
  h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
       color: var(--muted); font-weight: 600; margin: 2rem 0 .6rem; }
  .pos { color: var(--muted); font-style: italic; font-size: .9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>vocab</h1>
  <div class="sub"><span id="count">…</span> words in your deck<span id="model"></span></div>

  <form id="f">
    <input type="text" id="word" placeholder="a new word or phrase…" autofocus autocomplete="off">
    <select id="n">
      <option value="3" selected>3</option>
      <option value="5">5</option>
      <option value="8">8</option>
    </select>
    <button class="go" type="submit" id="go">Generate</button>
  </form>
  <div class="status" id="status"></div>
  <div id="out"></div>
</div>

<script>
const $ = s => document.querySelector(s);

fetch('/api/words').then(r => r.json()).then(d => { $('#count').textContent = d.count; });

function copyBtn(label, text) {
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = async () => {
    try {
      await navigator.clipboard.writeText(text);
      const old = b.textContent; b.textContent = 'copied';
      setTimeout(() => { b.textContent = old; }, 1200);
    } catch { b.textContent = 'copy failed'; }
  };
  return b;
}

function render(data) {
  const out = $('#out');
  out.textContent = '';

  data.candidates.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'card';

    const num = document.createElement('div');
    num.className = 'num';
    num.textContent = (i + 1) + (i === 0 ? '  ·  ' + data.part_of_speech : '');
    if (i === 0 && data.model) $('#model').textContent = '  ·  ' + data.model;
    card.appendChild(num);

    // front_html is built server-side and already escaped.
    const p = document.createElement('div');
    p.className = 'sentence';
    p.innerHTML = c.front_html;
    card.appendChild(p);

    const r = document.createElement('div');
    if (c.reused.length) { r.className = 'reuse'; r.textContent = 'reuses ' + c.reused.join(', '); }
    else { r.className = 'reuse none'; r.textContent = 'reuses nothing — stands on its own'; }
    card.appendChild(r);

    const row = document.createElement('div');
    row.className = 'row';
    row.appendChild(copyBtn('Copy front', c.front_html));
    row.appendChild(copyBtn('Copy back', data.back_html));
    card.appendChild(row);

    out.appendChild(card);
  });

  const h = document.createElement('h2');
  h.textContent = 'definition';
  out.appendChild(h);

  const dcard = document.createElement('div');
  dcard.className = 'card';
  const ul = document.createElement('ul');
  ul.className = 'defs';
  data.definition.forEach(d => {
    const li = document.createElement('li'); li.textContent = d; ul.appendChild(li);
  });
  dcard.appendChild(ul);
  const row = document.createElement('div');
  row.className = 'row';
  row.appendChild(copyBtn('Copy back HTML', data.back_html));
  dcard.appendChild(row);
  out.appendChild(dcard);
}

$('#f').onsubmit = async e => {
  e.preventDefault();
  const word = $('#word').value.trim();
  if (!word) return;
  $('#go').disabled = true;
  $('#status').textContent = 'writing sentences…';
  $('#out').textContent = '';
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({word, n: Number($('#n').value)}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'request failed');
    $('#status').textContent = '';
    render(data);
  } catch (err) {
    $('#status').textContent = String(err.message || err);
  } finally {
    $('#go').disabled = false;
  }
};
</script>
</body>
</html>
"""


def _handler(deck, profile, model=None, effort=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # quieter than the default
            pass

        def _send(self, code, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, payload) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/words":
                # Re-read every time: ~30 ms, so new cards show up without a restart.
                words = extract_terms(deck=deck, profile=profile)
                self._json(200, {"count": len(words), "words": words})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/generate":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(length) or b"{}")
                word = (req.get("word") or "").strip()
                if not word:
                    self._json(400, {"error": "no word given"})
                    return
                n = max(1, min(int(req.get("n") or 3), 10))

                words = extract_terms(deck=deck, profile=profile)
                from .generate import generate

                history = History.load()
                prefer, avoid = history.plan(words)
                result, _usage, used, used_effort = generate(
                    word,
                    words,
                    n=n,
                    model=model,
                    prefer=prefer,
                    avoid=avoid,
                    effort=effort,
                )
                known = {w.lower() for w in words}
                for cand in result.candidates:
                    history.record(verified_reuse(cand.sentence, cand.reused, known))
                history.save()

                self._json(
                    200,
                    {
                        "word": word,
                        "model": used + (f"/{used_effort}" if used_effort else ""),
                        "part_of_speech": result.part_of_speech,
                        "definition": result.definition,
                        "back_html": back_html(result.definition),
                        "candidates": [
                            {
                                "sentence": c.sentence,
                                "front_html": wrap_target(c.sentence, c.surface_form),
                                "reused": verified_reuse(c.sentence, c.reused, known),
                            }
                            for c in result.candidates
                        ],
                    },
                )
            except SystemExit as exc:  # missing credentials
                self._json(500, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(
    port: int = 8000,
    deck: str = "General",
    profile=None,
    open_browser: bool = False,
    model: str | None = None,
    effort: str | None = None,
) -> int:
    # 127.0.0.1, never 0.0.0.0: this is a personal tool with no authentication.
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(deck, profile, model, effort))
    except OSError as exc:
        print(f"Cannot bind port {port}: {exc}\nTry --port {port + 1}.")
        return 1

    url = f"http://127.0.0.1:{port}/"
    words = extract_terms(deck=deck, profile=profile)
    from .generate import resolve_model

    print(f"vocab · {len(words)} words from deck {deck!r} · {resolve_model(model)}")
    print(f"serving {url}  (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0
