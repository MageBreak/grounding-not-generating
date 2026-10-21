# Grounding, Not Generating

Mechanisms for an AI-assisted system-design pipeline that turns unstructured
business input into a structured product model **where every node and every
relationship resolves to a specific passage in the source**, and which refuses
rather than inventing when the evidence is not there.

> **Reason over evidence. Never invent evidence.**
>
> The system is meant to combine passages and resolve implicit relationships.
> It is not allowed to manufacture the evidence itself.

INT450 · Track B · SD-1 · SASTRA Deemed University
Bhuvan Mahesh · Aishwarya G · Mukundh M · mentor: Abhishek Pathak, Cisely.ai

---

## Headline findings

Measured on the AMI Meeting Corpus with **qwen3:14b run locally**, 34 scenario
trials, three instruction conditions, in a deterministic run and a 5-draw
sampled run that agree within a couple of points. Full tables and caveats in
[`docs/RESULTS.md`](docs/RESULTS.md).

**1 — Where the evidence exists, a string search is a perfect filter.**
In trials where all four roles are stated aloud, precision among assertions
whose quote resolved is **1.000** — across all three conditions and both runs,
170+ assertions without an exception. No gold labels and no model judgement
involved.

**2 — Where the evidence is absent, the same filter catches nothing.**
In trials where no role is stated, the quote test passes on ~90% of assertions
while ~60% of them are wrong. The model does not invent text; it reaches for
real nearby text and attaches it to a guess. `"Okay."` is a verified citation
and supports nothing. This is the measured case for an isolated audit stage.

**3 — True fabrication is 0.023, not the 0.120 it first appeared to be.**
81% of quotes that failed exact matching turned out to be the *right passage
with the disfluencies tidied away* — `uh`, `um`, `the the`, `<disfmarker>`.
Under the condition that forbids tidying, the same model reproduced them
verbatim. Invention and normalisation are different failures and must not
share a number.

**4 — Permission to refuse removes invention entirely, and costs coverage.**
Under `cite-or-refuse`, true fabrication is **0.000 in every tier**, precision
on partial-evidence trials is **1.000**, and coverage where nothing is stated
falls to 0.094. The price: it also refuses **half the time when the answer is
stated plainly**. Over-refusal is reported as prominently as fabrication.

**5 — Demanding a citation while forbidding refusal is the worst setting.**
It keeps coverage at 1.000 and raises fabrication above the no-instruction
baseline. Refusal is not an optional extra beside grounding; it is what makes
grounding possible.

---

## The pipeline

A language model appears at exactly two stages. Everything between and after
them is deterministic Python that can overrule it.
See [`docs/pipeline.html`](docs/pipeline.html) for the diagram.

| stage | what it does |
|---|---|
| 1 · retrieve | select candidate passages *(designed, not yet built)* |
| **2 · extract** | **model call** — schema forces a verbatim quote per claim |
| 3 · locate | quote → span, or nothing. A quote that cannot be found was not read |
| 4 · build | typed nodes and KG edges, provenance mandatory |
| 5 · validate | gates, ontology, path equations — no model involved |
| **6 · audit** | **model call** — a *different* model sees one span and one claim |
| 7 · escalate | violation → question. Never a null, never a guess |

Cross-cutting: every call and verdict is appended as JSONL, so a long run is
resumable; gold labels sit outside the repository and only the scorer reads
them.

---

## Layout

```
src/ami.py          AMI's NITE stand-off XML -> offset-addressable documents
src/schema.py       node types, the eight gates, escalation -- executable
src/ontology.py     UFO stereotypes; the gates, derived rather than listed
src/kg.py           edges with mandatory provenance + path-equation gates
src/kge.py          embeddings, constrained to suggesting and never asserting
src/llm.py          backend shim (Ollama / Gemini) + locate()
src/runner.py       the sweep: stages 1-7, resumable, never reads gold
src/audit.py        stage 6 -- isolated verifier, can only subtract
src/score.py        the only program that opens the answer key
src/relocate.py     separates paraphrase from fabrication
src/prototype.py    the regex baseline, kept for comparison
src/run.py          guided corpus tour
scripts/status.py   what is present on this machine and what has been run
docs/               corpus notes, schema, results, local-model setup, datasets
runs/               experiment output (JSONL)
data/               the AMI corpus (git-ignored -- fetched, never committed)
```

Every module runs standalone as its own self-test:

```bash
python3 src/llm.py        # locate() against fabricated and misplaced quotes
python3 src/ontology.py   # which gates fall out of which stereotype
python3 src/kg.py         # a graph where every node is perfect and the graph is not
python3 src/kge.py        # the scale check that refuses to overclaim
```

---

## Quick start

Python 3.11+, standard library only. Ollama for local inference —
see [`docs/LOCAL_MODEL.md`](docs/LOCAL_MODEL.md), especially the section on
Ollama's default context window, which **truncates silently** and would turn a
harness bug into what looks like a grounding failure.

```bash
git clone https://github.com/MageBreak/grounding-not-generating.git
cd grounding-not-generating

bash scripts/fetch_ami.sh                       # ~22 MB, not redistributed here
mkdir -p ~/gold && mv data/ami/corpusResources/meetings.xml ~/gold/

python3 scripts/status.py                       # verify the machine

ollama pull qwen3:14b
python3 src/runner.py --samples 1 --out runs/primary.jsonl
python3 src/score.py  runs/primary.jsonl
python3 src/relocate.py runs/primary.jsonl
```

---

## The gold-label wall

`corpusResources/meetings.xml` carries the authoritative speaker→role mapping.
**It is gold labels, not evidence** — nobody said it in a meeting; the
organisers recorded it.

It lives at `~/gold/meetings.xml`, outside the repository, so no `AMI_ROOT`
path can reach it and nothing browsing `data/` can stumble into it. Only
`score.py` opens it. `runner.py` and `audit.py` contain no reference to it, and
that is asserted rather than assumed.

This was not always true. `ami.load_meeting()` used to call `load_roles()`,
putting the answer key on every `Meeting` object handed to the extractors.
Nothing read it, so the published numbers stand — but the separation was a
comment rather than a property, and one careless line would have ended it.
Moving the file out is what exposed it.

---

## What this is not

There is no retrieval layer, no multi-agent extraction, and the audit stage
exists but has not been run at scale. The measurements cover **one model, one
corpus, one task** — a single field with four possible values, not a typed
model with relationships between entities. `docs/RESULTS.md` lists the limits
before anyone quotes a number.

## What comes next

1. Run the audit at scale and report `catch` / `falsecut`
2. A second and third model family — the cross-model claim needs them
3. Relationship-level extraction, where provenance has to cover edges
4. Move off AMI to a corpus with structural provenance
   (see [`docs/DATASETS.md`](docs/DATASETS.md))

---

## Data and licensing

The AMI Meeting Corpus is **not included here**. `scripts/fetch_ami.sh`
downloads it from the University of Edinburgh at run time into `data/`, which
is git-ignored. AMI is CC-BY-4.0; `runs/*.jsonl` contains short verbatim
excerpts under that licence. Cite:

> Carletta, J. et al. (2005). *The AMI Meeting Corpus: A Pre-Announcement.*
> Machine Learning for Multimodal Interaction.

Corpus home: https://groups.inf.ed.ac.uk/ami/corpus/
Code is MIT licensed — see `LICENSE`.
