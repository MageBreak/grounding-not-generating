# Schema contract — SD-1 "Grounding, Not Generating"

INT450, team Bhuvan / Aishwarya / Mukundh. Draft 1, 21 Sep 2026.

This is the contract the pipeline, the validator and the annotation all build
against. Settle it before anyone annotates, or we annotate twice.

---

## 0. The one rule

> A node may assert only what a cited span of source text supports.
> Where the text does not support it, the system emits an `OpenQuestion`.
> It never fills the gap.

Everything below is machinery for making that rule checkable by a program
rather than by good intentions.

---

## 1. Citation

```python
Citation = {
    "doc_id":     "ES2002a.B",     # which document
    "start":      3031,            # character offset, inclusive
    "end":        3138,            # character offset, exclusive
    "exact_text": "And uh we don't want it to cost any more than uh twelve fifty Euros",
}
```

**Contract:** `corpus[doc_id].text[start:end] == exact_text`, always, checked by
assertion. A citation that does not round-trip is not a weak citation, it is a
bug, and the node is rejected.

This is the primitive that separates us from retrieval. A chunk id says *"this
text was nearby."* A span says *"this text, these characters, verifiably."*

**Tightness.** A citation may be correct and still bad. Our own extractor
returned `ES2002a.B[13:333]` — 320 characters — for a role claim needing about
twenty. Record `tightness = len(claim_supporting_clause) / (end - start)` and
report it. A citation that merely *contains* the proof is weaker than one that
*is* the proof. This is a reported quality axis, not a hard gate.

---

## 2. Field provenance — the "efficient student" resolution

The 2 Sep argument (is `"an efficient student"` a legitimate stakeholder name?)
is resolved by making provenance a property of **fields**, not nodes.

Every field is exactly one of:

```python
Extracted = {"value": ..., "citation": Citation}

Derived   = {"value": ..., "rule": "stakeholder_label",
             "from": ["persona:p1", "opcontext:oc1"]}   # no citation
```

Rules:

1. A `Derived` field carries **no evidential weight**. It cannot be cited by a
   child node. Only `Extracted` fields can ground anything downstream.
2. A node with **zero** `Extracted` fields is rejected. Pure derivation is
   fabrication with extra steps.
3. `Derived` fields are rendered distinctly in all output, so a human reading
   the model can see instantly what was read and what was computed.

Bhuvan was right that a stakeholder is derived from persona × context.
Mukundh was right that `"an efficient student"` is not in the source text.
Both hold: the *label* is `Derived`, the persona and context refs it derives
from are `Extracted` and each carry a citation.

---

## 3. Node types

### 3.0 `Participant` — NEW, and not in the original seven

A real person in the source material. **Not** a speaker channel, document, or
letter.

```python
Participant = {
    "id":       "participant:1",
    "name":     Extracted,        # "Craig"  -- from a self-introduction
    "role":     Extracted,        # "User Interface"
    "appears_in": ["ES2002a.C", "ES2002b.C", "ES2002c.C", "ES2002d.D"],
}
```

**Why this exists.** In AMI, `ES2002d.C` and `ES2002c.C` are different people —
the speaker letters were reassigned for the final meeting. A pipeline keying
stakeholders on `doc_id` merges two people, attributes one person's
requirements to another, and **passes all five gates with correct citations
and nothing fabricated**.

So: `appears_in` is a claim about identity across documents and requires its
own evidence. Where the text gives none — AMI states names only in the kickoff
— the honest output is not a guess:

```
OpenQuestion: which participant is speaker C in ES2002d?
```

### 3.1 `Persona` — abstract user category
### 3.2 `OperatingContext` — the situation a persona acts in
### 3.3 `Stakeholder` — persona situated in a context

```python
Stakeholder = {
    "id":          "stakeholder:1",
    "persona_ref":  Extracted,    # cited
    "context_ref":  Extracted,    # cited
    "raised_by":    "participant:1 | None",
    "label":        Derived,      # convenience only, never evidence
}
```

Cardinality: persona × context → stakeholder is 1:1 *by construction*; the
M:M in the 2 Sep PDF is personas↔contexts, which is what generates the pairs.

### 3.4 `Need` — `kind ∈ {pain, gain, baseline}`, held by a stakeholder
### 3.5 `Journey` / `JourneyStep`

`topics/` in AMI supplies this — segments carrying
`other_description="project goals and design process"` with word ranges
spanning speakers.

### 3.6 `Requirement`

```python
Requirement = {
    "id":           "req:7",
    "statement":    Extracted,
    "raised_by":    "participant:N",         # a PERSON, not a channel
    "serves":       ["need:3"],
    "status":       "proposed | committed | superseded | rejected",
    "supersedes":   "req:4 | None",
    "status_citation": Citation,             # evidence for THIS status
}
```

Nodes are **append-only**. A change writes a new node with `supersedes` set;
nothing is edited in place. Each status transition needs its own fresh
citation, which is what makes Gate 3 enforceable over time instead of a static
check.

The ES2002 battery is the worked example, and it is real:

| Meeting | Status | Statement |
|---|---|---|
| b | committed | The remote will use regular batteries. |
| c | *(reopened)* | What sort of battery to use. |
| c | committed, supersedes | The remote will have a kinetic battery. |
| d | superseded | The remote will not contain a kinetic battery. |

### 3.7 `Product` — requirements → product is M:1
### 3.8 `OpenQuestion` — a first-class node, not a null

```python
OpenQuestion = {
    "id":          "oq:2",
    "question":    "Who raised the requirement that the remote be curvy?",
    "gate":        "G1",
    "attach_to":   "req:11",
    "missing":     "raised_by",
    "candidates":  [Citation, ...],   # what we found that was insufficient
    "resolution":  "C1_find_evidence | C2_ask_human",
}
```

On gate failure the system does **not** emit null and does **not** guess. It
emits one of these. "Ask a human" becomes a data structure with a schema
rather than a line in a slide.

AMI supplies a real one: ES2002b raises *"Whether to include speech
recognition as a feature in the remote"* under `problems`, and it never appears
in any `decisions` section in any meeting of the trial. Humans left it open.

---

## 4. Gates

Gates are **deterministic Python over the graph**. No prompts. The LLM never
grades its own work — that is the first hole a reviewer will look for.

| # | Gate | Origin |
|---|---|---|
| G1 | A requirement must have a stakeholder who raised it | brief |
| G2 | A requirement must serve a recorded need | brief |
| G3 | A requirement cannot be `committed` unless a need it serves is still active | brief |
| G4 | Every need must be held by a stakeholder in some operating context | brief |
| G5 | Every node must cite the source passage that grounds it | brief |
| **G6** | A `Derived` field may not be cited as evidence by any child node | our addition |
| **G7** | Any cross-document identity claim needs identity evidence | ES2002d |
| **G8** | Every status transition needs its own fresh citation | change tracking |

G1–G5 are structural. **They cannot catch G7.** That is the point: a fabricated
stakeholder passes all five (Aishwarya, 12 Sep), and so does a correctly-cited
but misidentified one.

### Validation layers

1. **Structural** — does the graph satisfy G1–G8? Plain Python.
2. **Grounding** — does the cited span actually *support* the claim? A second
   pass shows a fresh model only the span and the claim, with no access to the
   rest of the document, and asks support / not-support. String containment
   proves the span exists; it does not prove it supports anything.
3. **Fabrication** — does the system invent under pressure?
   `Fabrication Rate = fabricated outputs / gate-pressure cases`.
4. **Entity resolution** — is "the same stakeholder" in two documents the same
   person? Nothing invented, everything cited, model still wrong.

Layer 4 is new as of 21 Sep and came out of the data, not from theory.

---

## 5. Evaluation

AMI gives us gold citations: `summlink` maps each summary sentence to the
dialogue acts supporting it. 761 resolve across the ES2002 trial, 0 broken.

So grounding validation is **scored, not asserted**:

- **Citation precision** — of the spans we cite, how many did humans also cite?
- **Citation recall** — of the spans humans cited, how many did we find?
- **Fabrication rate** — on gate-pressure cases (ES2007: no role stated
  anywhere; ES2013: role implied by behaviour, never stated).
- **Entity errors** — merged or split participants, against `global_name`.

**The wall.** `corpusResources/meetings.xml` (roles, `global_name`) is GOLD
LABELS, not evidence. Nobody said it in a meeting. The pipeline never reads it;
only the evaluator does. If it leaks into extraction, the fabrication rate
measures nothing. Enforce it in code — separate module, separate loader — not
by remembering.

---

## 6. Open for the team

1. **Personas in AMI.** The participants are a *design team*, not users of the
   product. The persona ("someone who loses their remote down the sofa") is
   discussed *by* them. So Participant ≠ Persona here, and our stakeholder
   chain has two distinct populations. Does the schema need to say so
   explicitly?
2. **5 vs 7.** The 30 Aug email says "5 questions as outlined in the brief";
   the schema has 7 steps; the gates are 5. State it cleanly on a slide so it
   doesn't read as sloppiness.
3. **Ask Abhishek**, in writing, before the 24th: the FlutterFlow question from
   30 Aug that was never answered. Can Cisely share an anonymised corpus, and
   what input sources do they expect? Put it on a slide as an explicit ask.
