# vocab-gen

Generates Anki example sentences for a new vocabulary word that also reuse the words
already in your deck, so old vocabulary keeps resurfacing in new contexts.

Reads your collection directly. **Never writes to it.**

## Setup

```bash
cd ~/deuxcoast/vocab-gen
uv sync
```

Put your key in `.env` (already created, gitignored, mode 600):

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env.example` is the committed template. An exported `ANTHROPIC_API_KEY` in your shell
always wins over the file, so you can override per-run without editing anything. The key is
only needed for generating — `--list-words` works without one.

## Use

```bash
uv run vocab "sui generis"      # 3 candidate sentences + a draft definition
uv run vocab -n 5 obdurate      # more candidates
uv run vocab --html obdurate    # paste-ready HTML only
uv run vocab --list-words       # every vocab word it can see
uv run vocab --serve --open     # local web interface on 127.0.0.1:8000
uv run vocab --model haiku obdurate       # try a cheaper model
uv run vocab --usage --model sonnet obdurate   # ...and see what it cost
```

## Choosing a model and effort level

`--model` takes `opus`, `sonnet`, `haiku`, or any full model id. `--effort` takes `low`
(default), `medium`, `high`, `xhigh`, `max`. `VOCAB_MODEL` and `VOCAB_EFFORT` in `.env` set
persistent defaults; the flags override them.

Effort is the bigger lever, because output tokens dominate the bill and most of them are
invisible thinking. Measured on this prompt with Sonnet 5: **low 265 output tokens, medium
630, high 1,484** — low is roughly 4x cheaper than high, and the sentences held up.

Effort is dropped automatically for models that reject it (Haiku 4.5, Sonnet 4.5), rather
than failing the request.

Note that thinking counts against `max_tokens`, so the ceiling scales with effort. A budget
sized for `low` starves `high`: the model spends it reasoning and returns
`stop_reason=max_tokens` with no output at all.

At ~2,700 input tokens and ~1,200 output per card:

| Model | per card | per 1,000 cards |
|---|---|---|
| `opus` — claude-opus-5 | ~$0.044 | ~$44 |
| `sonnet` — claude-sonnet-5 | ~$0.017 | ~$17 |
| `haiku` — claude-haiku-4-5 | ~$0.009 | ~$9 |

The hard part of this prompt is restraint: reusing a known word *only* where it's natural, and
returning nothing when nothing fits. That is where smaller models tend to fail — they satisfy
the surface instruction and hand you a forced pairing. The second failure to watch for is
glossing the target word inside the sentence, which ruins the card as a recall test.

Because you pick from candidates, a weaker model costs your attention rather than your deck —
a bad sentence gets rejected and never lands. So it's worth running the same handful of words
through each and reading them side by side:

```bash
for m in haiku sonnet opus; do uv run vocab --model $m --usage perspicacious; done
```

The `reuses:` line is the thing to judge. If a model claims a pairing that reads as bolted
together, that's the tell.

The web interface is the nicer way to add several cards in a sitting: type a word, read the
candidates rendered as they'll look on the card, and hit **Copy front** / **Copy back**.

## How it finds your existing words

A vocab word is any text on a card's Front that is **both underlined and italicized**. On the
real collection that yields 767 unique terms and correctly skips the italic-only emphasis used
on non-vocab cards (`What does <i>brachii</i> mean?`).

Three details the extractor gets right, each of which was a real case in the collection:

- Terms are collected per *contiguous styled run*, not per note. One note underlining two
  unrelated words (`hilt`, `scabbard`) yields two terms, while a phrase inside one element
  (`sui generis`, `pumice stone`) stays whole.
- Both nesting orders (`<i><u>x</u></i>` and `<u><i>x</i></u>`) and `style`-attribute
  underline/italic are handled.
- Suspended cards are excluded.

## Staying current

There is no cache and no watcher. The full read — snapshot the collection, query, parse,
dedupe — takes about 30 ms, so it simply runs on every invocation and on every web request.
Words you added a minute ago are there; nothing to refresh or restart.

## Two things that will bite anyone reading an Anki collection

1. **`COLLATE unicase`.** Anki declares `notes.sfld`, `decks.name`, and `notetypes.name` with a
   custom collation. Plain `sqlite3` doesn't know it, and any query touching those columns
   fails with the unhelpful `OperationalError: no query solution`. `collection.py` registers a
   stub.
2. **The `-wal` file.** Anki keeps a write-ahead log next to the collection. Copying only
   `collection.anki2` silently gives you a stale snapshot missing recent edits — the snapshot
   copies `.anki2`, `-wal`, and `-shm` together.

## On sentence quality

The prompt's hardest instruction is that reusing nothing beats forcing a pairing; a sentence
that reads like two vocabulary words bolted together has failed even if both are used
correctly. Reuse claims are verified server-side — a word is only credited if it is genuinely
in your deck *and* genuinely present in the sentence — so the "reuses" line never overstates.

## Coverage: making the whole deck resurface

Every generation is an independent API call with no memory of the last one. Left alone, the
model re-expresses the same preference every time — it reaches for concrete, scene-building
nouns (`frigate`, `grove`, `isthmus`) because the prompt asks for vivid, specific prose, and
those are simply easier to write a scene around than abstract words. Measured over 36
sentences, only 3.2% of the deck was ever touched, and the abstract words that most need
re-exposure were the ones being skipped.

So the tool keeps a usage count per word in `~/.local/state/vocab-gen/usage.json` and feeds
two short lists into the *user* message: rarely-used words to favour, and just-used words to
skip. That message sits after the last cache breakpoint, so steering costs a few dozen tokens
and leaves the cached word list fully intact.

Measured with `frigate` and nine other words seeded as heavily used: **9 of 12 reuses came
from the rarely-used pool, 0 from the avoid list, and `frigate` appeared zero times.**

```bash
vocab --stats          # how much of the deck has actually appeared
vocab --no-history     # don't steer, don't record
```

Among equally-unused words the preferred set is *sampled*, not sliced alphabetically —
otherwise it would just trade one systematic bias for another.

Note that reuse is recorded for every candidate shown, not just the one you keep, since the
goal is variety in what you *see*.

## Tests

```bash
uv run pytest
```

## Notes

- Default port is 8000, not 8765: AnkiConnect already holds 8765.
- The server binds 127.0.0.1 only.
- Model is `claude-opus-5`. Your word list is sent as a cached prompt prefix; pass `--usage`
  to see actual token and cache numbers.
