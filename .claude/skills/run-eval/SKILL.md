---
name: run-eval
description: Set up, run, and read a vocab-gen eval comparison correctly. Use whenever comparing prompts, models, or selection strategies with scripts/eval.py, when interpreting a run's output, or when adding a metric or grader. Eval infrastructure fails by producing believable numbers rather than by crashing, and every rule here exists because it was broken once and the result looked fine.
---

# Running an eval comparison

The harness compares arms — a model paired with a prompt variant — over a fixed
40-word golden set, grades each candidate programmatically, and scores prose with
a blind comparative judge. It is easy to run in a way that produces a clean table
and means nothing. Each rule below was learned by shipping or nearly shipping a
wrong conclusion.

## Before running

**Put every arm in one run.** The judge scores candidates for a word side by
side, which is what makes it stable. Across runs the same model has drifted 0.25
judge points — larger than most effects worth measuring. An over-sampling
comparison was once run as two separate invocations and had to be discarded
entirely.

    uv run python scripts/eval.py moonshot --variants baseline balanced-reuse

**Freeze everything except the treatment.** Same cases, same seed, same model,
same judge. If the treatment *is* the prompt content — a different preferred-word
list, say — then that list must vary by arm, but keep the seed shared so the
difference is the bias and not the draw, and say in the commit that the
identical-prompt invariant was relaxed on purpose.

**Add a control when testing whether a mechanism works at all.** FSRS weighting
was compared against the score it replaced *and* against uniform sampling. It
beat uniform, which is the only reason we know the weighting does anything. The
old score, it turned out, was no better than not weighting at all.

**Never predict the outcome from stored runs.** This failed twice, in two
different ways. An intervention that changes how the data is produced cannot be
simulated from data produced the old way: asking one call for six candidates
changes the pool, and selecting for reuse changes which candidates you see. Pay
for the run.

## Before reading the output

**Check `err` per arm first, before any number.** An arm that lost cases has a
*biased* sample, not merely a small one — the survivors are whichever requests
happened to get through. Kimi once lost 14 of 20 cases to rate limiting and
reported confident scores from the remainder; a prompt variant lost 17 of 20 to a
content filter. Both tables looked fine.

If an arm is short, fix the cause and re-run the whole comparison. Do not patch
one arm and compare it against the others: they were judged without it.

## When reporting

**"noise" means the interval spans zero. It does not mean the arms are equal.**
A variant once read `naturalness -0.400, CI [-0.812, +0.012]` and was very nearly
shipped. At 40 words the same effect measured `-0.458, CI [-0.677, -0.239]` —
the estimate barely moved, the interval halved, and it was a real regression.

Quote the interval and the win/loss split, not just the verdict. Say "not
resolved at this sample size" rather than "no difference".

**Know the resolution floor.** Forty words resolves roughly ±0.2 judge points.
Arguing about a smaller difference is arguing about noise; double the golden set
instead (append, never reorder, so `subset(20)` still reproduces older runs).

**Scope the conclusion to what was tested.** A prompt that regressed Qwen by
0.458 cost Kimi nothing. Prompt effects do not reliably transfer between models,
so name the model in the claim.

## When adding a metric or grader

**Reuse the app's own checks.** A grader that reimplements matching will disagree
with what the tool actually does to a card. `unknown_claims` once tested literal
deck membership while verification was inflection-aware, so a claim of
`supplicant` against a deck holding `supplicants` counted as a hallucination —
inflating the metric for exactly the models that inflect.

**Grade what the user is shown.** With over-sampling the harness ranks and
truncates as the CLI does, rather than grading the raw pool, which no user ever
sees.

**Select by position, not by text.** `prepare()` strips markdown, so matching
candidates by sentence string silently drops any the model emphasised — shrinking
the sample rather than failing.

**Ask whether the metric can see the change at all.** FSRS altered *which words
are offered*; every existing metric only saw what happened to the sentence. The
`targeting` metric had to exist first or the work would have been unmeasurable.

## Reading the report

- `usable` — has the target word and invents no deck word
- `reuse` — fraction of sentences reusing at least one deck word
- `inv` — claimed a deck word that does not exist
- `targeting` — mean probability the learner had forgotten the reused words
- `$/100 acc` — dollars per *accepted* card; a cheap model needing more attempts
  is not cheap

Paired rows compare per word, which cancels word difficulty — the dominant source
of variance, and the reason 40 words can resolve anything at all.
