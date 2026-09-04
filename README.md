# vocab-gen

Generates Anki example sentences for a new vocabulary word that also reuse the words
already in your deck, so old vocabulary keeps resurfacing in new contexts.

Reads your collection directly. **Never writes to it.**

## Setup

```bash
cd ~/deuxcoast/vocab-gen
uv sync
```

spaCy and its `en_core_web_sm` model are pinned dependencies, so `uv sync` installs both. If
you also installed the CLI with `uv tool install`, **re-run it after any dependency change** —
the tool venv is separate from the project venv:

```bash
uv tool install --editable --force ~/deuxcoast/vocab-gen
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

## Providers

Models are addressed as `provider:model`; a bare id still means Anthropic, and a bare provider
name resolves to its default model.

```bash
vocab --model deepseek obdurate            # provider default
vocab --model zhipu:glm-4-plus obdurate    # explicit model
vocab --model ollama:qwen3 obdurate        # local, no key needed
```

Credentials come from the vendor's usual variable (`DEEPSEEK_API_KEY`, `ZHIPUAI_API_KEY`, ...)
or a `VOCAB_<PROVIDER>_API_KEY` override. Endpoints move, so
`VOCAB_<PROVIDER>_BASE_URL` overrides the built-in one.

`providers.py` exists because vendors disagree about the mechanics, not the task: how a system
prompt is passed, whether caching is explicit or automatic, whether a JSON *schema* can be
enforced or only "some JSON", and how reasoning depth is expressed. Providers that cannot
enforce a schema get it inlined in the prompt and their output recovered from markdown fences
or surrounding prose.

### Feasibility spike

Before comparing quality, check a provider can do the job at all:

```bash
uv run python scripts/spike.py                 # everything with credentials
uv run python scripts/spike.py deepseek zhipu  # specific ones
uv run python scripts/spike.py anthropic --repeat 2   # 2+ calls shows caching
```

It reports reachability, schema validity, tokens, cache behaviour, projected dollars per 100
cards, and — using the graders already in the app — how many reuse claims were verified versus
invented.

```
spec                           ok      in    out     cw     cr claim  ver  inv give  secs   $/100
anthropic:claude-sonnet-5      yes    980    300      0   5179     3    3    0    1   5.7   0.600
```

Rates in `MODEL_PRICING` were checked 2026-09-04 and will drift; the spike stamps the date it
is quoting. Endpoints and model ids drift too, which is why every one is overridable.

**Cheapest way to start:** `glm-4.7-flash` and `glm-4.5-flash` are free, and new Alibaba Cloud
accounts get a large free trial — enough to probe two vendors without paying anything.

One thing worth knowing before optimising for price: on EQ-Bench creative writing (Aug 2026)
Kimi K3 ranks second behind Claude Opus 5, ahead of GPT-5.6 — but it costs *more* per card
than Claude Sonnet 5. The cheap Chinese tier is 13-26x cheaper and, for this task, entirely
unmeasured. That gap is what the harness is for.

## Seeing failures

This is a single-user tool, so failures are shown in full rather than hidden behind a generic
message. Every failure is classified — `auth`, `balance`, `rate_limit`, `not_found`,
`connection`, `setup` — and reported with the provider, the model, and the exact environment
variable to check.

```bash
vocab --check     # probe every configured provider and report its status
```

The CLI prints failures to stderr in red; the web UI shows a red panel with the full message,
newlines intact.

If the default model cannot answer, the fallback runs — but never silently. A fallback prints
a warning naming what failed and what was used instead. If both fail, the message names both,
in order, so it points at the provider that actually broke first.

## Choosing a model and effort level

The default is **`dashscope:qwen3.8-flash`** with **`moonshot:kimi-k2.6`** as fallback. Over
234 judged candidates these were statistically indistinguishable from Claude Sonnet 5, at
$0.011 and $0.151 per 100 accepted cards against Sonnet's $0.293. An explicit `--model`
disables the fallback: choosing a model is an instruction, not a hint.

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

The web interface is the nicer way to add several cards in a sitting: type a word, pick a
candidate with **1-9** or by clicking it, and press **return** to send it straight to Anki —
deck `General`, note type `code-article`, tagged `vocab-gen` so anything the tool created can
be found or removed with a single search. Duplicates are reported rather than created, with a
**Send anyway** button for the legitimate second-sense case. **Copy front** / **Copy back**
are still there if you would rather paste by hand.

On the card front the target word is underlined and italic, and any deck words the sentence
reused are italic — so the connection is visible while reviewing.

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

Steering also uses what Anki knows about your recall. Each word carries a *shakiness* score
built from lapses and current interval, and the preferred pool is sampled weighted by it — so
a word you keep failing is likelier to resurface than one you have never missed. Measured over
24 sentences: reused words averaged **3.72 shakiness against a deck baseline of 2.45**.

When a candidate is generated, its reuse claims are checked **inflection-aware** (a deck entry
of `supplicants` is credited when the sentence writes `supplicant`) and recorded under the
deck's own spelling, so history aggregates one word to one key instead of scattering across
its forms.

Word matching uses spaCy lemmatization rather than suffix stripping, so `supplicant` matches a
deck entry of `supplicants` and `adumbrating` matches `adumbrated`, while derivation does not
collapse (`strait` never credits `straitjacket`). Part-of-speech tagging adds a check nothing
could do before: **if the card teaches the verb sense of `countenance` and the sentence uses
the noun, the card does not reinforce what was learned** — the candidate is flagged.

Candidates are also checked for **giving the answer away** — a sentence that hands over the
meaning turns a recall test into a freebie. Sharing a content word is not enough on its own:
a shared word counts when it is **rare enough to be informative** (Zipf below 4.5), or when
the sentence **reproduces at least half the definition's content words**, which is a
paraphrase however common the parts are.

The threshold came from the data rather than intuition. Across the flagged words in stored
runs, coincidental overlaps sat at Zipf 5.7–6.4 (`against`, `over`, `need`) and genuine
giveaways at 2.6–3.8 (`unyielding`, `bearers`, `treachery`). This matters beyond presentation:
`giveaway_rate` is one of the metrics prompts are compared on, so false positives add noise to
the measurement itself.

```bash
vocab --stats          # coverage, plus the shakiest words in your deck
vocab --no-history     # don't steer, don't record
```

After printing candidates the CLI asks which one you kept (skippable, TTY only); in the web UI
copying a candidate records it. Those kept sentences feed back in as calibration examples on
later runs.

Among equally-unused words the preferred set is *sampled*, not sliced alphabetically —
otherwise it would just trade one systematic bias for another.

Note that reuse is recorded for every candidate shown, not just the one you keep, since the
goal is variety in what you *see*.

## Evaluating prompts

The prompt is a bigger lever than the model — four models across three vendors landed within
noise of each other, which suggests the ceiling is set by the instructions, not the weights.

```bash
uv run python scripts/eval.py --list-variants
uv run python scripts/eval.py dashscope --variants baseline permissive-reuse terse
```

Variants live in `prompts.py` and are **composed from shared blocks**, not rewritten. An
ablation differs from the baseline in exactly the thing it claims to test; hand-rewriting a
whole prompt per variant is how a result gets attributed to the rule you changed on purpose
rather than the three words you changed by accident.

Every variant carries a stated **hypothesis**. If you cannot say in advance what it should do
to which metric, you are not running an experiment — and with enough variants, something
always wins by chance.

Comparisons are **paired**: every arm sees the same golden words, so the per-word difference
is averaged rather than each arm being averaged and subtracted. Word difficulty is the largest
source of variance — some targets are simply easier to write around — and pairing cancels it,
which is what lets 20 words resolve a difference at all.

```
  variant            metric            diff   95% ci     w/l  verdict
  terse              judge           -0.800    0.272   0/3    REAL
  terse              naturalness     -1.222    0.218   0/3    REAL
  terse              reuses/sent     +0.444    0.576   2/0    noise
```

"noise" means the interval spans zero — not that the arms are equal, only that this many words
cannot tell them apart.

## Eval harness

The spike answers "can this model do the job at all". The harness answers "which one is
better", which cannot be done by reading a few sentences — a prompt change earlier in this
project moved the reuse rate from 72% to 37% and eyeballing three sentences could not tell
whether that was an improvement.

```bash
uv run python scripts/eval.py                          # every ready provider
uv run python scripts/eval.py anthropic dashscope      # specific ones
uv run python scripts/eval.py --cases 5 --no-judge     # cheap smoke run
uv run python scripts/eval.py --report <run_id>        # re-render a stored run
```

Twenty target words, none of them in the deck, spread across part of speech and the
concrete/abstract axis — the axis where the model's word-selection bias showed up.

Most metrics are programmatic and reuse the app's own checks, so a grader can never disagree
with what the tool actually does to a card: does the sentence contain the target word, does it
invent deck words, does it leak the definition. The part no regex can score — *does this read
as written prose or as a vocabulary exercise* — goes to an LLM judge.

The judge is **blind and comparative**: candidates for one word are pooled across models,
shuffled, and labelled A/B/C, then scored side by side in one call. It never learns which
model wrote what. Naturalness is weighted double in the overall score.

**A caveat that cannot be engineered away:** judging Claude output with a Claude judge risks
self-preference. `--judge` is configurable for that reason; run it with judges from two
families before trusting a close result.

The headline number is **dollars per accepted card**, not per call. A model that is cheap per
request but needs more attempts before one is usable is not actually cheap.

Runs are stored as JSONL under `evals/runs/` so they can be re-analysed and compared without
paying to regenerate them.

## Tests

```bash
uv run pytest
```

## Notes

- Default port is 8000, not 8765: AnkiConnect already holds 8765.
- The server binds 127.0.0.1 only.
- Model is `claude-opus-5`. Your word list is sent as a cached prompt prefix; pass `--usage`
  to see actual token and cache numbers.
