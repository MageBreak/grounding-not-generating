# Corpus notes — AMI, what it gives us and what it doesn't

INT450 SD-1 "Grounding, Not Generating". Status as of 21 Sep 2026.

## What we pulled

Source: `github.com/gcunhase/AMICorpusXML` (the Edinburgh AMI site and Hugging Face
are both blocked from our build environment, so this mirror is what we have).

- 171 meeting transcripts
- 687 speaker-separated transcripts (`<meeting>.<speaker>.transcript.txt`)
- 142 abstractive + 137 extractive human summaries
- 35 scenario trials (ES/IS/TS), each 4 meetings: kickoff → functional →
  conceptual → detailed design

Built into `corpus/<trial>.docs.json` — `{doc_id: text}` where `doc_id` is
`<meeting>.<speaker>`, e.g. `ES2002a.C`. A citation is `(doc_id, start, end)`
and `text[start:end]` must equal the quoted span. Verified by assertion.

## What AMI gives us for free

**Grounded stakeholders.** Speakers state their own roles in the kickoff, in
citable text: *"Um I'm Craig and I'm User Interface."* (`ES2002a.C[0:36]`).
Gate 1 becomes checkable against real evidence, not a synthetic label.

**Roles are NOT in a fixed order.** ES2002 is A=Industrial Designer,
B=Project Manager, C=User Interface, D=Marketing Expert. The conventional
PM/ME/UI/ID assumption is wrong here. Any system that assumes the order is
fabricating.

**Longitudinal structure.** Meetings a/b/c/d in one trial, so a requirement
proposed in `b` and dropped in `c` is in the data. Makes
`proposed → committed → superseded` demonstrable rather than theoretical.

**Naturally occurring evidence gaps.** Only **6 of 35 trials** have all four
roles stated in the kickoff text. We do not have to manufacture the gap —
the corpus already contains it.

## The four cases we selected

| Trial | Role evidence | What it tests |
|---|---|---|
| `ES2002` | 4/4 stated | Control. Happy path, everything grounded. |
| `ES2013` | 0/4 stated | Implicit role. A runs the kickoff — *"Welcome everyone. Um this is the kick-off meeting for the day"* — behaves as PM, never says the title. |
| `ES2007` | 0/4 stated | Zero evidence. Nobody states a role at all. |
| `ES2003` | attribution trap | B (the PM) does a roll call: *"And your role is? You're the Marketing Expert."* C says *"I'm the Marketing Expert."* The string sits in B's document but refers to C. |

`ES2013` is the "efficient student" argument from 2 Sep appearing in real data —
strong contextual inference, no explicit evidence. `ES2003` is the case that
separates our two validation layers: string containment passes (the span exists,
it's in the document) but the span does not support the claim *about that
speaker*. Structural gates cannot catch it; grounding verification can.

Because AMI's scenario design guarantees all four roles exist in every trial, a
model under pressure on `ES2007` will confidently emit PM/ME/UI/ID — plausible,
structurally perfect, ungrounded, and (given ES2002's ordering) probably assigned
to the wrong speakers. That is the fabrication experiment, on real data, with
known ground truth.

## What the full annotation set added (21 Sep, after pulling ami_public_manual_1.6.2)

The Edinburgh zip (21.8 MB) changed the picture. Working copy of ES2002 is in
`/home/claude/ami/es2002`; resolver is `src/ami.py`.

**Turn interleaving is back.** `segments/` gives utterances with
`transcriber_start`/`transcriber_end`; merging four speakers by timestamp
reconstructs the real conversation. Limitation 1 below is retired for any
trial we pull properly.

**Gold citations exist.** `extractive/<meeting>.summlink.xml` links each
abstractive summary sentence to the dialogue acts that support it. Chain:
summary claim → dialogue act → word id range → character offsets. On ES2002a,
**59 of 59 citations resolve with zero breaks**. Humans annotated which
utterances support which claim, so grounding validation can be scored as
precision/recall against human annotators rather than asserted.

**Requirement evolution is annotated.** ES2002 battery decisions run:
`b` regular batteries → `c` (problem reopened) → `c` kinetic → `d` not kinetic.
LCD and buttons do the same. Free `proposed → committed → superseded` chains.

**A permanently open question.** ES2002b raises *"Whether to include speech
recognition as a feature in the remote"* under `problems`. It never appears in
any `decisions` section in any meeting of the trial. A naturally occurring
`OpenQuestion`, annotated as unresolved by humans.

**Journey layer.** `topics/` gives segments with `other_description` such as
"introduction of participants and their roles", spanning speakers by word range.
Schema step 6 goes from weakest to annotated.

**Dialogue acts map to our vocabulary.** `Suggest` is where requirements get
proposed; `Elicit-Inform` is someone asking a question — the escalation path,
pre-labelled. ES2002a: 164 Inform, 77 Assess, 25 Elicit-Inform, 12 Suggest.

**Role extraction validated.** Our span-grounded extraction (A=ID, B=PM, C=UI,
D=ME, read only from self-introductions) matches `meetings.xml` ground truth
exactly for meetings a, b and c — no metadata consulted.

### The ES2002d finding — a failure class we had not accounted for

`meetings.xml` appears to say C and D swap roles in the final meeting:

    ES2002a/b/c:  A=ID  B=PM  C=UI  D=ME
    ES2002d:      A=ID  B=PM  C=ME  D=UI

They do not. The `global_name` attribute settles it — the people keep their
roles throughout:

    MEE006 = ID    FEE005 = PM    MEE007 = UI    MEE008 = ME

What changed is which **letter** each person was assigned. In a/b/c,
C=MEE007 (Craig, UI) and D=MEE008 (Andrew, ME). In `d` they are exchanged.
Talk volume confirms it — Craig is the quiet one in every meeting
(541, 3922, 4903, 3947 chars) and Andrew the verbose one
(5744, 11981, 15309, 14340). The volumes follow the person, not the letter.

**Why this matters more than the battery chain.** `ES2002d.C` and `ES2002c.C`
are different people. Any pipeline that treats the speaker letter as a stable
entity key across documents — the obvious thing to do — silently merges two
people into one stakeholder and attributes Andrew's requirements to Craig.

All five gates pass. Every citation is *correct*: the span exists, it is in
that document, that speaker did say it. Nothing is fabricated. The extraction
is right and the model is still wrong, because identity resolution across
documents is wrong.

That is a fourth failure class, beside Aishwarya's three:

1. Structural — does it obey the rules?
2. Grounding — is each node supported by its cited evidence?
3. Fabrication — does it invent under pressure?
4. **Entity resolution — is "the same stakeholder" in two documents the same
   person?** Nothing invented, everything cited, model still wrong.

The honest handling is also grounded: names are only stated in meeting `a`
("I'm Craig"), so in meeting `d` the identity of speaker C **cannot be
established from text alone**. The correct output is not a guess but an
`OpenQuestion`: *which participant is speaker C in ES2002d?* Exactly the
escalation path, arising from real data rather than a constructed example.

## Limitations — what AMI does NOT give us

1. ~~**No turn interleaving.**~~ RETIRED — `segments/` restores it.
2. ~~**No timings, no dialogue-act annotations.**~~ RETIRED — `words/` carries
   `starttime`/`endtime`, `dialogueActs/` carries typed acts.
3. **Speech, not documents.** Disfluencies throughout (`Um`, `T_V_`, `Mm-hmm`).
   Genuinely messy, which suits the brief — but offsets must point into the raw
   text, never a cleaned copy, or citations stop being verifiable.
4. **One domain only.** Every scenario trial is the same TV-remote design task.
   Good for depth, bad for showing the pipeline generalises.
5. **No email or documents**, despite the AMI papers describing collected email
   and slides — this mirror does not carry them. So our corpus is
   single-modality, while the brief describes multi-source input.
6. **No gold product model.** Nobody has annotated AMI with
   personas/needs/requirements. We build that ourselves, by hand, and it is the
   bottleneck.

## What we still need for a complete PoC

- [x] ~~**Hand-annotated gold graph**~~ — largely SOLVED. `abstractive/` supplies
      decisions/actions/problems per meeting; `summlink` supplies their citations.
      Remaining hand work is mapping those onto our seven-step schema
      (persona / operating context / need), which is far smaller than annotating
      from scratch.
- [ ] **Requirement-level ablation pairs.** Role gaps come free; requirement gaps
      do not. Clone `ES2002`, delete the spans grounding 3–4 chosen requirements,
      keep both versions.
- [ ] **A naive baseline extractor.** The control condition. Without it
      "our system doesn't fabricate" is unfalsifiable.
- [ ] **A second domain** so the pipeline isn't shown on one task only. Cheapest
      option: Python PEPs or Rust RFCs — a named proposer, a stated motivation,
      explicit rejected alternatives, and public.
- [ ] **A multi-source artifact set** (email chain, ticket dump) to match the
      brief's "internal documentation, meeting minutes, etc." Small and
      hand-written is fine.
- [ ] **Answer from Abhishek on real input data** — the FlutterFlow question
      from 30 Aug that was never answered. Ask in writing before 24 Sep: can
      Cisely share an anonymised corpus, and what input sources do they expect?
      Put it on a slide as an explicit ask.

## Licensing

AMI is CC-BY-4.0. Cite Carletta et al. 2005. Fine for coursework and for showing
Abhishek; do not redistribute the transcripts in the deck beyond short quoted
spans (which is all our citations are anyway).
