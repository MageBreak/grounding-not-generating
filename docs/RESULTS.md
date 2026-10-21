# Results

Everything here is a **prototype / baseline** result on the AMI Meeting Corpus.
AMI is a controlled environment for proving mechanisms, not the deployment
case. Read `CORPUS_NOTES.md` for what it does and does not give us.

---

## The task

For each of 34 AMI scenario trials, show a model the kickoff meeting and ask
who holds which of four roles (PM / ME / UI / ID), demanding a verbatim quote
and the document it came from for every answer.

Trials are labelled by **how much evidence actually exists**, computed from the
transcript by the regex baseline's own extractor and never from gold:

| tier | meaning | trials |
|---|---|---:|
| `full` | all four roles stated aloud | 7 |
| `partial` | one to three stated | 19 |
| `none` | no role stated by anyone | 8 |

This gradient occurs naturally in AMI. We did not construct it.

## The conditions

The output schema is identical in all three — one slot per speaker, each with
`role`, `doc_id`, `quote`, where `role` may be `not_stated`. Only the
instruction changes, so the instruction is the only variable.

| condition | instruction |
|---|---|
| `p0_complete` | answer for every speaker; do not leave anyone unassigned |
| `p1_cite` | give the quote that states it; copy exactly |
| `p2_cite_or_refuse` | as `p1`, and `not_stated` is a correct answer |

## Runs

| file | model | temp | draws | cells |
|---|---|---:|---:|---:|
| `runs/primary.jsonl` | qwen3:14b | 0.0 | 1 | 102 |
| `runs/variance.jsonl` | qwen3:14b | 0.7 | 5 | 510 |

Zero failed calls in either. Reasoning mode off and recorded as a condition.

---

## 1. Fabrication: 0.023, not 0.120

An assertion whose quote cannot be found in the document it was attributed to
looked like fabrication. Inspecting ES2012 showed that most of it is not.

Same model, same trial, different instruction:

```
p0_complete : "Right welcome to the first meeting of Real Reaction's development meetings…"
p1_cite     : "Right w welcome to the the first meeting of uh Real Reaction's uh um development meetings…"
transcript  : "Right w welcome to the the first meeting of uh Real Reaction's uh um development meetings…"
```

Under `p0` the model silently cleaned the transcription — dropped a false
start, a stutter, two fillers. Under `p1`, which forbids tidying, it
reproduced every disfluency and the quote located exactly. **It had found the
right passage both times.**

`src/relocate.py` re-examines failed matches after normalising inline markup
(`<disfmarker>`, `<other>`), filler (`uh`, `um`, `mm`) and immediate
repetitions, then falls back to a sequence match at ratio ≥ 0.70.

| run | failed exact | recovered | genuinely absent | apparent FAB | **true FAB** |
|---|---:|---:|---:|---:|---:|
| primary | 35 | 27 (77%) | 8 | 0.110 | **0.025** |
| variance | 189 | 153 (81%) | 36 | 0.120 | **0.023** |

True fabrication by tier (variance run):

| condition | full | partial | none |
|---|---:|---:|---:|
| `p0_complete` | 0.000 | 0.039 | 0.000 |
| `p1_cite` | 0.000 | 0.047 | 0.019 |
| `p2_cite_or_refuse` | **0.000** | **0.000** | **0.000** |

Two things fall out:

- **Zero genuine fabrication wherever the evidence exists**, under every
  condition, in both runs.
- **Zero genuine fabrication under `p2` in every tier.** Permitting refusal
  removes invention entirely.
- What remains concentrates in `partial` — the tier where *some* evidence
  exists, which is precisely where filling the gap is tempting.

A recovered quote is still not a pass. The model did not copy what it was
asked to copy, and a citation that cannot be verified byte for byte is worth
less than one that can. But it is not invention, and the two must not share a
number.

---

## 2. Where evidence exists, a string search is a perfect filter

`gprec` is precision among assertions whose quote resolved.

| condition | tier | asserted | located | prec | **gprec** |
|---|---|---:|---:|---:|---:|
| `p0_complete` | full | 28 | 26 | 0.929 | **1.000** |
| `p1_cite` | full | 28 | 25 | 0.893 | **1.000** |
| `p2_cite_or_refuse` | full | 15 | 13 | 0.867 | **1.000** |

Over 170 assertions across both runs, in the `full` tier, **a located quote
has never once been attached to a wrong answer.** No gold labels, no model
judgement — a string search separates right from wrong without error.

## 3. Where evidence is absent, the same filter catches nothing

| condition | tier | prec | gprec | FAB (exact) |
|---|---|---:|---:|---:|
| `p0_complete` | none | 0.375 | 0.414 | 0.094 |
| `p1_cite` | none | 0.438 | 0.483 | 0.094 |

In the `none` tier the quote test passes on ~90% of assertions while ~60% of
them are wrong. The model does not invent text; it reaches for real nearby
text and attaches it to a guess. One example, located and verified and
supporting nothing:

```
claim: speaker B is the project manager
quote: "Okay."            (occurs dozens of times in the document)
```

**This is the empirical case for the isolated audit stage**, and it is a
measurement rather than an argument. `locate()` is necessary and not
sufficient, and its blind spot sits exactly in the tier that matters.

`gprec` degrades monotonically with evidence availability:

```
full 1.000  →  partial 0.787  →  none 0.436        (p0_complete, variance)
```

## 4. Permission to refuse works, and it costs

| condition | tier | prec | cov |
|---|---|---:|---:|
| `p0_complete` | partial | 0.692 | 1.000 |
| `p2_cite_or_refuse` | partial | **1.000** | 0.355 |
| `p0_complete` | none | 0.357 | 1.000 |
| `p2_cite_or_refuse` | none | 0.667 | **0.094** |
| `p2_cite_or_refuse` | full | 0.933 | **0.536** |

On `partial`, permitting refusal takes precision to **1.000** — never wrong
when it answers. On `none`, coverage falls to **0.094**: it declines almost
everything where nothing is stated, which is the behaviour the brief asks for.

**The cost is the `full` row.** Coverage 0.536 means it refuses about half the
time when the answer is stated plainly in front of it. That is over-refusal,
and it belongs in the results as prominently as the fabrication rate. A system
that will not answer a question the transcript plainly answers is not safe; it
is useless.

## 5. Asking for a citation without permitting refusal is the worst setting

`p1_cite` keeps coverage at 1.000 and raises apparent fabrication to 0.145 on
`partial` — higher than `p0_complete`. Demanding a citation while forbidding
refusal makes the model manufacture citations. Refusal is not a nice-to-have
that sits beside grounding; it is what makes grounding possible.

## 6. The speaker-letter trap, in model output

30 quotes (variance) were real text located in a **different document** than
the one claimed. That is the same failure as `ES2002d.C` vs `ES2002c.C` — a
speaker label treated as a stable identity — now appearing in model output
rather than in our own schema. Counted separately as `misplaced`.

---

## Comparison to the regex baseline

`prototype.py`, same task, no model:

| extractor | precision | coverage | fabrication |
|---|---:|---:|---:|
| grounded (cites or refuses) | 0.970 | 0.493 | — |
| naive (always answers) | 0.610 | 1.000 | 0.507 |

The hand-written grounded extractor and `p2_cite_or_refuse` land in the same
region of the precision/coverage plane, from opposite directions. The naive
baseline's 0.507 was a *simulation* of fabrication via a hardcoded prior; the
real model's true fabrication rate is 0.023. **The simulation overstated the
problem by roughly 20×** — which is worth saying plainly, because we published
that number first.

---

## Reproduce

```bash
bash scripts/fetch_ami.sh
mkdir -p ~/gold && mv data/ami/corpusResources/meetings.xml ~/gold/

ollama pull qwen3:14b
python3 src/runner.py --samples 1 --out runs/primary.jsonl
python3 src/score.py runs/primary.jsonl
python3 src/relocate.py runs/primary.jsonl
```

## What these numbers are not

- **One model.** qwen3:14b only. "Models fabricate at rate X" needs more than
  one family; this says "qwen3:14b fabricates at rate X on this task".
- **One corpus.** Four-person scripted role-play, not scattered business
  material.
- **One task.** A single field with four possible values, not a typed model
  with relationships between entities.
- **Not yet audited.** Stage 6 exists (`src/audit.py`) but has not been run at
  scale. Until it has, the `none`-tier hole is measured and not closed.
