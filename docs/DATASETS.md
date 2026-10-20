# Datasets

## The selection principle

Most corpora are useless to us and it is worth being precise about why.

Summarisation corpora give you (document, summary) pairs. We cannot score
fabrication against a summary, because a summary that happens to be right tells
you nothing about whether the model *read* the passage or *knew* the answer.

What we need is **span-level ground truth**: a record of which passage supports
which claim. That is rare. AMI has it (`extractive/summlink.xml` links summary
sentences to dialogue acts, which resolve to word ranges, which resolve to
character offsets). Almost nothing else does.

So the catalogue below is ordered by how close each corpus gets to that, not by
size or fame. Three strategies, in descending order of value:

1. **Corpora with annotated provenance** — rare, small, directly scoreable.
2. **Corpora where provenance exists structurally** — an RFC that was produced
   by a mailing list thread, a commit linked to an issue. Ground truth comes
   free from the link structure rather than from annotation.
3. **Corpora with no provenance** — useful only for qualitative work, or after
   you annotate a slice yourself.

Strategy 2 is where the project should go after AMI. It is the only way to get
realistic, genuinely unstructured input at scale without an annotation budget.

---

## 1. Span-level provenance (strategy 1)

### AMI Meeting Corpus — *have it*
What you already use. 100h of scenario-driven meetings, NITE stand-off
annotation, human extractive-summary links down to word ranges.
- Manual annotations ~22 MB, CC-BY-4.0.
- <https://groups.inf.ed.ac.uk/ami/corpus/> · `scripts/fetch_ami.sh`
- Limits recorded in `CORPUS_NOTES.md`. Chief one: it is scripted role-play
  with four participants, not scattered business material.

### ICSI Meeting Corpus
AMI's sibling: ~70 real research-group meetings, same NITE annotation lineage,
including dialogue acts and extractive summaries. Naturally occurring rather
than scenario-driven, so the evidence is messier — which is the point.
- The obvious second corpus, because your `ami.py` resolver mostly transfers.
- <https://groups.inf.ed.ac.uk/ami/icsi/>

### RAGTruth
Word-level hallucination annotations over ~18k responses, across summarisation,
QA and data-to-text, with each unsupported span marked and typed.
- The closest public thing to our own metric, and the right external comparison
  for a fabrication-rate number.
- Niu et al., ACL 2024 · <https://arxiv.org/abs/2401.00396>

---

## 2. Requirements engineering — the actual target domain

### PURE — PUblic REquirements dataset ★
79 real natural-language requirements documents, ~34k sentences, collected from
the web and offered both as raw documents and in a common XML schema.
- **The single best fit for the deployment case.** These are the artifacts our
  pipeline is supposed to produce, written by people, in the wild.
- No span-level provenance, so the natural use is the *inverse* task: given the
  requirements document, test whether the model can resolve each requirement to
  the stakeholder, need and context it implies — and refuse where it can't.
- Ferrari et al., RE 2017 · <https://zenodo.org/record/1414116>

### PROMISE NFR dataset
625 labelled requirements (functional + 11 non-functional categories). Small,
clean, decades old, and still the default classification baseline.
- Use for the schema-conformance half, not the grounding half.

### CrowdRE / app-review requirement datasets
User feedback and app reviews labelled with requirement-bearing content. Closest
public analogue to "unstructured stakeholder input" at volume.
- Noisy in the way real input is noisy. Good stress test for refusal behaviour.

---

## 3. More meetings and multi-party discussion

### ELITR Minuting Corpus
~120 hours of meetings in English and Czech with **multiple human-written
minutes per meeting** — several annotators, same source.
- Valuable for a reason that is easy to miss: disagreement between annotators is
  a measurement of how much of a "correct" model is genuinely determined by the
  evidence. If two humans extract different requirements from one meeting, that
  is the ceiling on what we can demand of a model.
- CC-BY-NC-SA · <https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4692>

### MeetingBank
1,366 city-council meetings, 3,579 hours, transcripts with agendas and
summaries, and segment-level structure.
- Real institutional decision-making with a document trail. The agenda/minutes
  pairing gives weak structural provenance.
- Hu et al., ACL 2023 · <https://arxiv.org/abs/2305.17529> ·
  <https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio> ·
  <https://github.com/YebowenHu/MeetingBank-utils>

### QMSum
Query-based meeting summarisation over AMI, ICSI and committee meetings, with
annotated **relevant text spans per query**.
- Spans per query is close to what we want. Worth a look before ICSI, since it
  partly annotates the corpus you already parse.

---

## 4. Provenance that exists structurally (strategy 2) — where to go next

None of these are packaged datasets. All of them are better than packaged
datasets, because the link between unstructured discussion and the resulting
specification is recorded by the community itself.

| Source | Unstructured input | The "model" it produced | Link |
|---|---|---|---|
| **IETF** | working-group mailing lists | RFCs | RFCs cite discussion; drafts are versioned |
| **Python** | discuss.python.org threads | PEPs | PEP header links the discussion thread |
| **Apache** | JIRA issues + dev@ lists | release notes, commits | issue IDs in commit messages |
| **Rust** | internals.rust-lang.org, RFC PRs | accepted RFCs | RFC PR contains the whole debate |
| **Kubernetes** | SIG meeting notes, GitHub issues | KEPs | KEP references its issues |

Why this is the right move: you get genuinely unstructured, multi-month,
multi-author, contradictory discussion, paired with an authoritative structured
outcome, with the provenance edges already recorded. Requirements change,
get superseded, and are raised by people who then disappear — all the things
AMI's four-person scripted meetings cannot give you.

**Pick one project and go deep.** Rust RFCs or PEPs are the best starting
point: moderate volume, excellent discussion-to-document linkage, permissive
licensing, and a stable API to fetch from.

The annotation you'd still need is small: for ~100 requirements, mark which
passage in the thread actually supports it. That is a weekend across four
people, and it is the most valuable thing the team could produce.

---

## 5. Grounding and attribution benchmarks

For comparing your fabrication rate against published numbers, and for borrowing
evaluation design rather than inventing it.

- **ALCE** — LLMs with citations; measures citation precision/recall against
  retrieved passages. The closest published methodology to ours.
- **HAGRID** — generative IR with human-annotated attribution.
- **ExpertQA** — expert-curated questions with attribution judgements.
- **FActScore** — decomposes a generation into atomic facts and scores each.
  The decomposition idea transfers directly to scoring a product model node by
  node instead of as a whole.

Use these to position the work, not to train on. Our task — building a typed,
related model rather than answering a question — is different enough that the
numbers are not directly comparable, and you should say so rather than implying
they are.

---

## 6. Ontology and KG resources

### OntoUML/UFO Catalog ★
A FAIR, curated catalogue of **ontology-driven conceptual models** in OntoUML,
with stereotypes, serialised as machine-readable graphs, plus a Python library
for querying them.
- Directly feeds `src/ontology.py`. Before inventing stereotype assignments for
  our types, look at how a few hundred published models assign theirs.
- <https://zenodo.org/records/8188545> · <https://github.com/unibz-core> ·
  `pip install ontouml-models-lib`

### gUFO / UFO
The foundational ontology behind the stereotypes in `ontology.py`. gUFO is the
lightweight OWL implementation, suitable if you ever want reasoner support.
- Caution already noted in the code: OWL is open-world and our refusal semantics
  are closed-world. "Not stated" is a *finding* for us and merely "unknown" to a
  reasoner. Do not let a reasoner quietly convert one into the other.

### ConceptNet / Wikidata / WordNet
General-purpose background knowledge. Relevant for entity linking and for
disambiguating mentions across documents — the ES2002 failure class.
- Treat anything they supply the way `kge.py` treats embeddings: a candidate
  generator, never a grounding source. A fact from Wikidata is not a fact
  anybody in your corpus said.

---

## 7. KGE benchmarks — for testing `kge.py`, not for results

Our graph is far too small for embeddings to mean anything (`assess_scale()`
will tell you so). But you still need to know the pipeline works before you
trust it as a candidate generator. Run it on a standard benchmark first:

- **FB15k-237** — 310k triples, the default sanity check.
- **WN18RR** — 93k triples, harder, no inverse-relation leakage.
- **CoDEx** (s/m/l) — Wikidata-derived, with hard negatives.
- **OGB link-prediction tasks** — if you want a larger-scale comparison.

All ship with PyKEEN. If RotatE on FB15k-237 reproduces roughly the published
MRR, your wrapper is correct; if it doesn't, nothing you compute on the project
graph is interpretable.

---

## 8. The normalised shape, and how to get there

Everything above arrives in a different format. Convert once, to this, and the
harness never has to care again:

```jsonl
# corpus.jsonl — one line per document
{"doc_id": "pep-0572", "source": "pep", "text": "...", "meta": {"year": 2018}}

# units.jsonl — optional segmentation (turns, sentences, paragraphs)
{"doc_id": "pep-0572", "start": 0, "end": 184, "kind": "paragraph", "speaker": null}

# gold.jsonl — only where real provenance exists
{"doc_id": "pep-0572", "claim": "assignment expressions are opt-in",
 "span": [1204, 1281], "label": "supported"}
```

The one invariant, same as everywhere else in this project:

```python
corpus[doc_id][start:end] == exact_text
```

Assert it at conversion time. A converter that silently shifts offsets by one
will produce a fabrication rate that is entirely an artifact of your own
pipeline, and it will look plausible.

**Where a model helps, and where it must not.** Use Gemini or a local model to
do the boring reshaping — pulling text out of HTML, splitting threads into
messages, normalising speaker labels, guessing a document's date. Those are
transformations you can verify by inspection.

Do not use a model to produce `gold.jsonl`. Gold labels generated by a language
model and then used to score a language model measure agreement, not
correctness, and the error is invisible in every downstream number. Gold comes
from human annotation or from structural links the community already recorded —
nowhere else. The same wall that keeps `meetings.xml` out of the extraction path
applies to anything you generate to replace it.
