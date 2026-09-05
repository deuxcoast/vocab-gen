"""A small local web interface for composing cards.

Read-only: it reads the Anki collection and never writes to it. Bound to
127.0.0.1 so it is not reachable from the network.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collection import extract_vocab
from . import anki
from .generate import GenerationError
from .history import History
from .render import back_html, prepare, rank_candidates

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
    padding: 1.1rem 1.2rem; margin-bottom: .9rem; cursor: pointer;
    position: relative; transition: border-color .12s, box-shadow .12s;
  }
  .card:hover { border-color: var(--muted); }
  .card.selected {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 7%, var(--card));
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 30%, transparent);
  }
  .card.selected::before {
    content: ''; position: absolute; left: 0; top: 12%; bottom: 12%;
    width: 3px; border-radius: 3px; background: var(--accent);
  }
  .pick {
    position: absolute; top: .9rem; right: 1rem; font: .72rem ui-monospace, monospace;
    color: var(--muted); border: 1px solid var(--line); border-radius: 4px;
    padding: .05rem .35rem;
  }
  .card.selected .pick { color: var(--accent); border-color: var(--accent); }
  .sendbar {
    display: flex; align-items: center; gap: .7rem; margin: 1.2rem 0 .4rem;
    position: sticky; bottom: 0; z-index: 5;
    background: var(--bg); padding: .7rem 0;
    border-top: 1px solid var(--line);
  }
  .sendbar .selinfo { font-size: .82rem; color: var(--accent); font-weight: 600; }
  .sendbar button {
    background: var(--accent); border-color: var(--accent); color: #fff;
    font-weight: 600; padding: .55rem 1rem;
  }
  .sendbar .hint { color: var(--muted); font-size: .82rem; }
  .sent { color: var(--ok); font-size: .85rem; }
  .num { color: var(--muted); font-size: .8rem; font-family: ui-monospace, monospace; }
  .sentence { margin: .35rem 0 .8rem; }
  .sentence u { text-underline-offset: 3px; }
  .reuse { font-size: .82rem; color: var(--ok); font-family: ui-monospace, monospace; }
  .reuse.none { color: var(--muted); }
  .giveaway { font-size: .82rem; color: #b4462f; font-family: ui-monospace, monospace; margin-top: .2rem; }
  .alert { border: 1px solid #b4462f; border-radius: 10px; padding: .9rem 1.1rem;
           margin-bottom: 1rem; background: color-mix(in srgb, #b4462f 8%, transparent); }
  .alert h3 { margin: 0 0 .35rem; font-size: .95rem; color: #b4462f; }
  .alert pre { margin: 0; white-space: pre-wrap; font: .82rem/1.5 ui-monospace, monospace;
               color: var(--fg); }
  .alert.warn { border-color: #a9761f; background: color-mix(in srgb, #a9761f 8%, transparent); }
  .alert.warn h3 { color: #a9761f; }
  @media (prefers-color-scheme: dark) { .giveaway { color: #e08a70; } }
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
  <div id="out" tabindex="-1"></div>
</div>

<script>
const $ = s => document.querySelector(s);
let lastErrorTitle = '';
let selected = null;
let current = null;

function select(i) {
  selected = i;
  let target = null;
  document.querySelectorAll('.card[data-idx]').forEach(el => {
    const on = Number(el.dataset.idx) === i;
    el.classList.toggle('selected', on);
    if (on) target = el;
  });
  const bar = $('#sendbar');
  if (bar) bar.hidden = (i === null);
  const label = $('#selinfo');
  if (label) label.textContent = (i === null) ? '' : `candidate ${i + 1} selected`;
  // Without this you can press a number, highlight a card below the fold, and
  // see nothing happen — which reads as the key not working. 'nearest' is a
  // no-op when the card is already on screen, so this needs no visibility test
  // of its own; hand-rolling one only invents a way to get it wrong.
  // Instant, not smooth: smooth scrolling is a no-op in some contexts, and a
  // keyboard picker wants the card there before the next keystroke anyway.
  if (target) target.scrollIntoView({block: 'nearest'});
}

async function sendSelected(allowDuplicate) {
  if (selected === null || !current) return;
  const c = current.candidates[selected];
  const btn = $('#send');
  btn.disabled = true;
  $('#sendmsg').textContent = 'sending…';
  try {
    const res = await fetch('/api/send', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        front: c.front_html, back: current.back_html, word: current.word,
        sentence: c.sentence, reused: c.reused, allow_duplicate: !!allowDuplicate,
      }),
    });
    const data = await res.json();
    if (res.status === 409 && data.duplicate) {
      $('#sendmsg').innerHTML = '';
      const warn = document.createElement('span');
      warn.className = 'hint';
      warn.textContent = data.error + ' ';
      const again = document.createElement('button');
      again.textContent = 'Send anyway';
      again.onclick = () => sendSelected(true);
      $('#sendmsg').appendChild(warn); $('#sendmsg').appendChild(again);
      return;
    }
    if (!res.ok) throw new Error(data.error || 'send failed');
    $('#sendmsg').innerHTML = '';
    const ok = document.createElement('span');
    ok.className = 'sent';
    ok.textContent = `added to ${data.deck}, tagged ${data.tag}`;
    $('#sendmsg').appendChild(ok);
  } catch (err) {
    $('#sendmsg').textContent = '';
    showAlert('error', 'Could not add the card', String(err.message || err));
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener('keydown', e => {
  const el = e.target;
  const tag = (el && el.tagName) || '';
  const typing =
    tag === 'TEXTAREA' ||
    (tag === 'INPUT' && !['checkbox', 'radio', 'button', 'submit'].includes(el.type));
  if (typing) return;
  if (e.key >= '1' && e.key <= '9') {
    const i = Number(e.key) - 1;
    if (current && i < current.candidates.length) { select(i); e.preventDefault(); }
  } else if (e.key === 'Enter' && selected !== null) {
    sendSelected(false); e.preventDefault();
  }
});
const KIND = {
  auth: 'API key rejected', balance: 'out of credit', rate_limit: 'rate limited',
  not_found: 'no such model', connection: 'cannot reach the API', other: 'failed',
};

function showAlert(level, title, body) {
  const box = document.createElement('div');
  box.className = 'alert' + (level === 'warn' ? ' warn' : '');
  const h = document.createElement('h3');
  h.textContent = title;
  const pre = document.createElement('pre');
  pre.textContent = body;
  box.appendChild(h); box.appendChild(pre);
  $('#out').prepend(box);
}

fetch('/api/words').then(r => r.json()).then(d => { $('#count').textContent = d.count; });

function copyBtn(label, text, onCopy) {
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = async () => {
    try {
      await navigator.clipboard.writeText(text);
      if (onCopy) onCopy();
      const old = b.textContent; b.textContent = 'copied';
      setTimeout(() => { b.textContent = old; }, 1200);
    } catch { b.textContent = 'copy failed'; }
  };
  return b;
}

function render(data) {
  const out = $('#out');
  out.textContent = '';
  current = data; selected = null;
  window.scrollTo({top: 0});
  // Take focus off whatever form control has it and give it to the results.
  // Blurring only the word field was not enough: submitting with return leaves
  // focus in the field, and the count select swallows number keys too. With a
  // definite home for keystrokes the shortcuts do not depend on how the
  // generation happened to be triggered.
  if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
  setTimeout(() => out.focus({preventScroll: true}), 0);

  data.candidates.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.idx = i;
    card.onclick = ev => {
      if (ev.target.tagName === 'BUTTON') return;
      // Clicking a card means the user is choosing, not typing.
      if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
      select(i);
    };

    const pick = document.createElement('span');
    pick.className = 'pick';
    pick.textContent = i + 1;
    card.appendChild(pick);

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

    if (c.missing_target) {
      const m = document.createElement('div');
      m.className = 'giveaway';
      m.textContent = 'does not use the target word at all — unusable';
      card.appendChild(m);
    }
    if (c.wrong_sense) {
      const w = document.createElement('div');
      w.className = 'giveaway';
      w.textContent = 'used as ' + c.wrong_sense + ', not the sense on the card';
      card.appendChild(w);
    }
    if (c.giveaway && c.giveaway.length) {
      const g = document.createElement('div');
      g.className = 'giveaway';
      g.textContent = 'gives the answer away: ' + c.giveaway.join(', ');
      card.appendChild(g);
    }

    const row = document.createElement('div');
    row.className = 'row';
    row.appendChild(copyBtn('Copy front', c.front_html, () => {
      fetch('/api/choose', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({word: data.word, sentence: c.sentence, reused: c.reused})}).catch(()=>{});
    }));
    row.appendChild(copyBtn('Copy back', data.back_html));
    card.appendChild(row);

    out.appendChild(card);
  });

  const bar = document.createElement('div');
  bar.className = 'sendbar';
  bar.id = 'sendbar';
  bar.hidden = true;
  const send = document.createElement('button');
  send.id = 'send';
  send.textContent = 'Send to Anki';
  send.onclick = () => sendSelected(false);
  const info = document.createElement('span');
  info.className = 'selinfo';
  info.id = 'selinfo';
  const hint = document.createElement('span');
  hint.className = 'hint';
  hint.textContent = 'press 1-9 to pick, return to send';
  const msg = document.createElement('span');
  msg.id = 'sendmsg';
  bar.appendChild(send); bar.appendChild(info); bar.appendChild(hint); bar.appendChild(msg);
  out.appendChild(bar);

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
    if (!res.ok) {
      const who = data.title || data.provider;
      lastErrorTitle = who
        ? `${who} — ${data.also ? 'both failed' : (KIND[data.kind] || data.kind || 'failed')}`
        : 'Request failed';
      throw new Error(data.error || 'request failed');
    }
    $('#status').textContent = '';
    if (data.fell_back_from) {
      const f = data.fell_back_from;
      showAlert('warn', `${f.provider} — ${KIND[f.kind] || f.kind}; used ${data.model} instead`, f.message);
    }
    render(data);
  } catch (err) {
    $('#status').textContent = '';
    showAlert('error', lastErrorTitle || 'Request failed', String(err.message || err));
  } finally {
    $('#go').disabled = false;
  }
};
</script>
</body>
</html>
"""


def _handler(deck, profile, model=None, effort=None, oversample=1):
    def load_vocab():
        v = extract_vocab(deck=deck, profile=profile)
        return v, [w.term for w in v]

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
                _, words = load_vocab()
                self._json(200, {"count": len(words), "words": words})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/api/send":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    req = json.loads(self.rfile.read(length) or b"{}")
                    front = (req.get("front") or "").strip()
                    back = (req.get("back") or "").strip()
                    if not front:
                        self._json(400, {"error": "nothing to send"})
                        return
                    allow = bool(req.get("allow_duplicate"))
                    if not allow and anki.is_duplicate(front, back):
                        self._json(409, {
                            "duplicate": True,
                            "error": "Anki already has a card with this front.",
                        })
                        return
                    note_id = anki.add_note(front, back, allow_duplicate=allow)
                    # Keeping it is the strongest signal that a candidate was good.
                    history = History.load()
                    history.record_kept(
                        (req.get("word") or "").strip(),
                        (req.get("sentence") or "").strip(),
                        list(req.get("reused") or []),
                    )
                    history.save()
                    self._json(200, {"ok": True, "note_id": note_id,
                                     "deck": anki.DECK, "tag": anki.TAG})
                except anki.AnkiError as exc:
                    self._json(502, {"error": str(exc)})
                except Exception as exc:
                    self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if self.path == "/api/choose":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    req = json.loads(self.rfile.read(length) or b"{}")
                    history = History.load()
                    history.record_kept(
                        (req.get("word") or "").strip(),
                        (req.get("sentence") or "").strip(),
                        list(req.get("reused") or []),
                    )
                    history.save()
                    self._json(200, {"ok": True})
                except Exception as exc:
                    self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
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

                vocab, words = load_vocab()
                from .generate import generate

                history = History.load()
                prefer, avoid = history.plan(vocab)
                outcome = generate(
                    word,
                    words,
                    n=n,
                    model=model,
                    prefer=prefer,
                    avoid=avoid,
                    effort=effort,
                    kept=history.recent_kept(),
                    oversample=oversample,
                )
                result = outcome.result
                used, used_effort = outcome.model, outcome.effort
                cands = rank_candidates(prepare(result, word, words), n)
                for c in cands:
                    history.record(c["reused"])
                history.save()

                self._json(
                    200,
                    {
                        "word": word,
                        "model": used + (f"/{used_effort}" if used_effort else ""),
                        "fell_back_from": (
                            outcome.fell_back_from.as_dict()
                            if outcome.fell_back_from is not None
                            else None
                        ),
                        "part_of_speech": result.part_of_speech,
                        "definition": result.definition,
                        "back_html": back_html(result.definition),
                        "candidates": cands,
                    },
                )
            except GenerationError as exc:
                # Surfaced in full: this is a single-user tool, and a hidden
                # failure is worse than a visible one.
                self._json(502, {"error": exc.message, **exc.as_dict()})
            except SystemExit as exc:
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
    oversample: int = 1,
) -> int:
    # 127.0.0.1, never 0.0.0.0: this is a personal tool with no authentication.
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(deck, profile, model, effort, oversample))
    except OSError as exc:
        print(f"Cannot bind port {port}: {exc}\nTry --port {port + 1}.")
        return 1

    url = f"http://127.0.0.1:{port}/"
    words = [w.term for w in extract_vocab(deck=deck, profile=profile)]
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
