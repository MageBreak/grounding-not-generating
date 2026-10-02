# Grounding, Not Generating — AMI prototype

Prototype mechanisms for an AI-assisted system-design pipeline that turns
unstructured business input into a structured product model **where every node
resolves to a specific passage in the source corpus**, and which refuses rather
than inventing when the evidence is not there.

> **Reason over evidence. Never invent evidence.**
>
> The system is meant to combine passages and resolve implicit relationships.
> It is not allowed to manufacture the evidence itself.

INT450 · Track B · SD-1 · SASTRA Deemed University
Bhuvan Mahesh · Aishwarya G · Mukundh M · mentor: Abhishek Pathak, Cisely.ai

---

## What this repository is — and is not

**It is** a set of verified primitives, exercised against the AMI Meeting
Corpus as a controlled environment: citation resolution, a deterministic
validator, and a fabrication-pressure experiment.

**It is not** the full pipeline. There is no LLM in this code. Extraction here
is pattern matching, used deliberately as a baseline. The multi-agent
extraction, the evidence-search loop and the grounding audit are designed and
not yet implemented.

Every number below is a **prototype / baseline** result. Read `docs/RESULTS.md`
before quoting any of it.

---

## Results

Across 34 AMI scenario trials, 136 speaker-role assertions, scored against gold
labels that neither extractor is permitted to read:

| Extractor | Asserted | Correct | Precision | Coverage |
|---|---:|---:|---:|---:|
| **Grounded** — cites or refuses | 67 | 65 | **0.970** | 0.493 |
| **Naive** — always answers | 136 | 83 | **0.610** | 1.000 |

**Fabrication rate = 69/136 = 0.507.** Of those 69 unevidenced claims, **51
were wrong — 73.9%**.

Separately, the citation resolver walks all 761 human-annotated citations in the
ES2002 trial down to exact character spans with **zero broken chains**.

The naive extractor is better on every structural measure — complete, confident,
all five gates green. It is wrong 39% of the time, and nothing in its output
separates the 61% it got right from the 39% it invented.

---

## Quick start

Python 3.11+. **No third-party dependencies** — standard library only.

```bash
git clone https://github.com/<you>/grounding-not-generating.git
cd grounding-not-generating

# fetch the AMI annotations (~22 MB, not redistributed here)
bash scripts/fetch_ami.sh          # Windows: powershell scripts/fetch_ami.ps1

python src/run.py                  # corpus tour + findings
python src/schema.py               # validator self-test
python src/prototype.py            # the fabrication experiment
```

Each script takes an optional path to the unzipped annotations; by default they
look in `data/ami/`.

---

## Layout

```
src/ami.py          resolve AMI's NITE stand-off XML into offset-addressable documents
src/schema.py       the node types, the gates, and escalation — as executable code
src/prototype.py    grounded vs naive extraction, scored against gold labels
src/run.py          a guided tour of the corpus that reproduces every finding
docs/               corpus notes, the schema contract, the experiment record
data/               the AMI corpus (git-ignored — fetched, never committed)
```

### `ami.py` — the resolver

AMI uses NITE stand-off annotation: text lives in `words/`, and every other
layer points at ranges of it by word id. Nothing is readable until the pointers
resolve. This module renders each speaker's words into one string **while
recording every token's character offset in the same pass**, so text and offsets
cannot drift, then resolves segments, dialogue acts and gold citations against it.

The contract is one line, and it is asserted:

```python
corpus[doc_id][start:end] == exact_text
```

A citation that does not round-trip is not a weak citation. It is a bug, and the
node is rejected.

### `schema.py` — the validator

Node types, the extracted/derived field split, and the gates as **deterministic
Python over the graph**. No prompts, no model call, nothing that asks an LLM
whether its own output was acceptable.

The five gates are the brief's. Three more came out of failures we found in the
data — see `docs/SCHEMA.md`. `escalate()` turns each violation into an
`OpenQuestion` rather than a null or a guess.

Run it directly for a self-test on a real failure case from ES2002.

### `prototype.py` — the experiment

Two extractors over the same text, same task. The grounded one asserts a role
only where a speaker states it about themselves in a citable sentence-level
span. The naive one always answers, falling back on the scenario prior where
evidence is missing — standing in for a language model's world knowledge.

The output ends with the grounded extractor's **own** residual errors, printed
with their spans. Those are left in deliberately: they are the test set for the
grounding-verification layer, and they are the clearest demonstration that
string containment is not support.

---

## The gold-label wall

`corpusResources/meetings.xml` carries the authoritative speaker→role mapping
and a `global_name` per participant. **This is gold labels, not evidence** —
nobody said it in a meeting; the organisers recorded it.

The extractors never read it. Only the scorer does. If that mapping leaks into
extraction, a trial where no role is stated anywhere stops being a fabrication
test — the system already knows the answer, and the fabrication rate measures
nothing.

The separation is kept by `load_gold()` living in the scorer, not in `ami.py`'s
extraction path. If you extend this code, keep it that way.

---

## Data and licensing

The AMI Meeting Corpus is **not included in this repository**. `scripts/fetch_ami.sh`
downloads it from the University of Edinburgh at run time into `data/`, which is
git-ignored.

AMI is distributed under CC-BY-4.0. If you use it, cite:

> Carletta, J. et al. (2005). *The AMI Meeting Corpus: A Pre-Announcement.*
> Machine Learning for Multimodal Interaction.

Corpus home: https://groups.inf.ed.ac.uk/ami/corpus/

The code in this repository is MIT licensed — see `LICENSE`.

---

## What comes next

AMI is a controlled environment for proving out mechanisms, not the target
problem. The deployment case is genuinely unstructured material — evidence
scattered across documents, stakeholders mentioned once in passing, implicit
relationships, contradictions, and requirements that change.

Planned, in order:

1. Evidence and retrieval layer over an unstructured corpus
2. Model-driven extraction with verifiable quoting (`locate`)
3. Grounding verification — an isolated pass that sees only the span and the claim
4. The full evidence-search and human-question loop
5. Evaluation: fabrication rate, grounding fidelity, question quality, gap
   detection, expert edit distance

`docs/CORPUS_NOTES.md` records what AMI gives us and what it does not.
