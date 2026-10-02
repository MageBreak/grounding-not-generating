# Fabrication experiment — results

INT450 SD-1. Run 21 Sep 2026 over all 34 AMI scenario trials (kickoff meeting
of each), 136 speaker-role assertions. `src/prototype.py`.

## Setup

Two extractors, same text, same task: name each speaker's role.

- **Grounded** — asserts a role only where a speaker states it about
  themselves in a citable sentence-level span. Where the evidence is absent it
  emits an `OpenQuestion` instead of a guess.
- **Naive** — always returns a complete answer. Where evidence is missing it
  falls back on the scenario prior (PM/ME/UI/ID by speaker order). This is what
  a language model does under gate pressure.

Scored against `corpusResources/meetings.xml`. **Neither extractor reads it.**
Only the scorer does. If that wall leaks, the fabrication rate measures nothing.

## Headline

|  | asserted | correct | precision | coverage |
|---|---|---|---|---|
| **Grounded** | 67 | 65 | **0.970** | 0.493 |
| **Naive** | 136 | 83 | **0.610** | 1.000 |

**Fabrication rate = 69/136 = 0.507.**
Of those 69 unevidenced claims, **51 were wrong — 73.9%**.

The naive extractor is better on every structural measure. Complete, confident,
all five gates green, no nulls. It is wrong 39% of the time, and nothing in the
output distinguishes the 61% it got right from the 39% it invented.

The grounded extractor answers half as often and is right 97% of the time when
it does. It says "I don't know" 69 times.

**That the invented answers are wrong 73.9% of the time is the finding.** The
common intuition is that a model filling a gap makes a *plausible* guess that is
often roughly right. It does not. It is wrong nearly three times in four, and
the wrong answer is structurally indistinguishable from the right one.

## Iteration — what the errors taught us

First run: precision 0.915, six errors. Categorised:

| Error | Class | Fixable? |
|---|---|---|
| IS1009a C — "my role is industrial designer … technical-functional designs" read as UI | pattern-list ordering bug | yes |
| IS1000a D — "You are the marketing guy? … So you are the marketing" | second-person attribution | yes |
| ES2009a A — 333-char span | loose window | yes |
| TS3011a A — 759-char span from offset 633 | loose window | yes |
| TS3012a A — 195-char span from offset 6397 | loose window | yes |
| IS1003a D — "trends … converged to the User" | **semantic** | no |

Three changes, each traceable to a specific error:

1. **Cite the sentence, not the utterance**, capped at 240 characters. Refuse
   rather than cite loosely.
2. **Earliest role mention in the text wins**, not first in the pattern list.
3. **Reject second person** — `you're`, `you are`, `your role`, `he's`, `she's`.

Second run: precision **0.970**, coverage 0.493 (down from 0.522), same 65
correct answers from 4 fewer assertions. Coverage fell *by design* — the span
cap makes the extractor refuse cases it previously guessed at.

## The two survivors

Both are tight, first-person, single-sentence spans. Both wrong. Neither is
fixable by pattern discipline.

**IS1003a speaker D** — extracted UI, gold ME:

> *"Yeah, I'm just go I'm trying to give you some trends about what should be
> done and what the users would like to have and then thi this would converged
> to the User [Interface]"*

D is the Marketing Expert describing marketing work ("trends") being handed
*to* the UI designer. The span contains `I'm` and contains `User Interface`.
String containment passes. The span does not support the claim.

**TS3012a speaker A** — extracted ME, gold PM:

> *"Marketing &lt;gap&gt; I'm not sure what you what you came up with yet."*

A is the Project Manager *addressing* the Marketing Expert. 64 characters,
first person, no second-person marker to filter on.

Both require knowing **who the role term refers to**, not whether it is
present. That is semantics, and it is precisely what validation layer 2 exists
for — a second pass showing a fresh model only the span and the claim, with no
access to the rest of the document, asking support / not-support.

## What to say on the 24th

The first run had six errors. Four were our own sloppiness and we fixed them.
Two were not, and those two define the architecture: structural gates cannot
catch a correct-looking citation that does not support its claim. That is why
grounding validation is a separate layer and not a substring check.

A result that reports its own residual failure mode is more credible than a
clean number, and it pre-empts the obvious question — *what catches the cases
your gates can't see?*

## Caveats

- Role extraction is **one** slice of the schema. It is the one with gold
  labels for all 34 trials, which is why the experiment runs here.
- The naive baseline's prior (PM/ME/UI/ID by speaker order) is a stand-in for
  a language model's scenario knowledge. A real LLM baseline would be a
  stronger control and is the obvious next step.
- Coverage of 0.493 is not a target. Roughly half of AMI's kickoffs simply do
  not contain self-stated roles; the honest ceiling is set by the corpus.
