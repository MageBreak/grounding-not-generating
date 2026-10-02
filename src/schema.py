"""Executable schema contract for SD-1 "Grounding, Not Generating".

Companion to SCHEMA.md. The gates are deterministic Python over the graph --
no prompts, no model involvement. The LLM never grades its own work.

Run directly for a self-test:  python3 schema.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Citation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Citation:
    doc_id: str
    start: int
    end: int
    exact_text: str

    def verify(self, corpus: dict[str, str]) -> bool:
        """The contract: the span must round-trip against the raw document."""
        doc = corpus.get(self.doc_id)
        return doc is not None and doc[self.start:self.end] == self.exact_text

    def tightness(self, supporting_clause: str) -> float:
        """How much of the cited span actually does the work. Reported, not gated."""
        span = self.end - self.start
        return len(supporting_clause) / span if span else 0.0

    def __repr__(self):
        return f"{self.doc_id}[{self.start}:{self.end}]"


# --------------------------------------------------------------------------- #
# Field provenance -- the "efficient student" resolution
# --------------------------------------------------------------------------- #


@dataclass
class Extracted:
    """Read from the source. Carries evidence. May ground children."""
    value: Any
    citation: Citation


@dataclass
class Derived:
    """Computed from parent nodes. NO evidential weight. May NOT ground children."""
    value: Any
    rule: str
    frm: list[str] = field(default_factory=list)


Field = Extracted | Derived


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


@dataclass
class Node:
    id: str
    fields: dict[str, Field] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.id.split(":")[0]

    def extracted(self) -> dict[str, Extracted]:
        return {k: v for k, v in self.fields.items() if isinstance(v, Extracted)}

    def derived(self) -> dict[str, Derived]:
        return {k: v for k, v in self.fields.items() if isinstance(v, Derived)}

    def citations(self) -> list[Citation]:
        return [f.citation for f in self.extracted().values()]


@dataclass
class Participant(Node):
    """A real person. NOT a speaker channel or document id.

    appears_in is a cross-document identity claim and needs its own evidence
    (see G7). ES2002d.C and ES2002c.C are different people.
    """
    appears_in: list[str] = field(default_factory=list)
    identity_evidence: dict[str, Citation] = field(default_factory=dict)


@dataclass
class Requirement(Node):
    raised_by: str | None = None          # participant id -- a PERSON
    serves: list[str] = field(default_factory=list)   # need ids
    status: Literal["proposed", "committed", "superseded", "rejected"] = "proposed"
    supersedes: str | None = None
    status_citation: Citation | None = None


@dataclass
class Need(Node):
    kind: Literal["pain", "gain", "baseline"] = "pain"
    held_by: str | None = None            # stakeholder id
    active: bool = True


@dataclass
class Stakeholder(Node):
    persona_ref: str | None = None
    context_ref: str | None = None
    raised_by: str | None = None          # participant id


@dataclass
class OpenQuestion:
    id: str
    question: str
    gate: str
    attach_to: str
    missing: str
    candidates: list[Citation] = field(default_factory=list)
    resolution: Literal["C1_find_evidence", "C2_ask_human"] = "C2_ask_human"


@dataclass
class Model:
    """The product model: nodes, plus the questions we refused to answer."""
    nodes: dict[str, Node] = field(default_factory=dict)
    open_questions: list[OpenQuestion] = field(default_factory=list)

    def add(self, n: Node) -> Node:
        self.nodes[n.id] = n
        return n

    def of_kind(self, kind: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == kind]


# --------------------------------------------------------------------------- #
# Gates -- deterministic, no model involvement
# --------------------------------------------------------------------------- #


@dataclass
class Violation:
    gate: str
    node_id: str
    detail: str

    def __repr__(self):
        return f"[{self.gate}] {self.node_id}: {self.detail}"


def check(model: Model, corpus: dict[str, str]) -> list[Violation]:
    v: list[Violation] = []

    for n in model.nodes.values():
        # G5 -- every node must cite the passage that grounds it
        if not n.citations():
            v.append(Violation("G5", n.id, "no citation on any field"))
        for name, f in n.extracted().items():
            if not f.citation.verify(corpus):
                v.append(Violation("G5", n.id,
                                   f"field '{name}' citation does not round-trip "
                                   f"({f.citation})"))
        # a node of pure derivation is fabrication with extra steps
        if n.derived() and not n.extracted():
            v.append(Violation("G6", n.id, "every field is Derived -- nothing grounds this"))

        # G7 -- cross-document identity claims need identity evidence
        if isinstance(n, Participant) and len(n.appears_in) > 1:
            for doc in n.appears_in:
                if doc not in n.identity_evidence:
                    v.append(Violation(
                        "G7", n.id,
                        f"claims to appear in {doc} with no identity evidence -- "
                        f"speaker channels are not stable across documents"))

    for r in model.of_kind("req"):
        assert isinstance(r, Requirement)
        # G1 -- a requirement must have a stakeholder who raised it
        if not r.raised_by:
            v.append(Violation("G1", r.id, "no participant raised this"))
        elif r.raised_by not in model.nodes:
            v.append(Violation("G1", r.id, f"raised_by {r.raised_by} does not exist"))
        # G2 -- a requirement must serve a recorded need
        if not r.serves:
            v.append(Violation("G2", r.id, "serves no recorded need"))
        for nid in r.serves:
            if nid not in model.nodes:
                v.append(Violation("G2", r.id, f"serves {nid} which does not exist"))
        # G3 -- committed requires a still-active need
        if r.status == "committed":
            live = [model.nodes[x] for x in r.serves if x in model.nodes]
            if not any(getattr(x, "active", False) for x in live):
                v.append(Violation("G3", r.id,
                                   "marked committed but no need it serves is active"))
        # G8 -- status transitions need their own fresh citation
        if r.status in ("committed", "superseded", "rejected"):
            if r.status_citation is None:
                v.append(Violation("G8", r.id,
                                   f"status '{r.status}' has no status_citation"))
            elif not r.status_citation.verify(corpus):
                v.append(Violation("G8", r.id, "status_citation does not round-trip"))
        if r.supersedes and r.supersedes not in model.nodes:
            v.append(Violation("G8", r.id, f"supersedes {r.supersedes} which does not exist"))

    for nd in model.of_kind("need"):
        assert isinstance(nd, Need)
        # G4 -- every need must be held by a stakeholder in some operating context
        if not nd.held_by:
            v.append(Violation("G4", nd.id, "held by no stakeholder"))
        elif nd.held_by not in model.nodes:
            v.append(Violation("G4", nd.id, f"held_by {nd.held_by} does not exist"))
        else:
            sh = model.nodes[nd.held_by]
            if isinstance(sh, Stakeholder) and not sh.context_ref:
                v.append(Violation("G4", nd.id,
                                   "stakeholder has no operating context"))

    return v


def escalate(violations: list[Violation]) -> list[OpenQuestion]:
    """Gate failure never yields null and never yields a guess."""
    ASK = {
        "G1": ("Who raised this requirement?", "raised_by"),
        "G2": ("Which recorded need does this requirement serve?", "serves"),
        "G3": ("Is a need this requirement serves still active?", "status"),
        "G4": ("Which stakeholder holds this need, in what context?", "held_by"),
        "G5": ("Which source passage grounds this node?", "citation"),
        "G6": ("What evidence supports this, beyond derivation?", "extracted"),
        "G7": ("Is this the same person across both documents?", "appears_in"),
        "G8": ("What evidence supports this status change?", "status_citation"),
    }
    out = []
    for i, x in enumerate(violations, 1):
        q, miss = ASK.get(x.gate, ("What evidence supports this?", "unknown"))
        out.append(OpenQuestion(id=f"oq:{i}", question=q, gate=x.gate,
                                attach_to=x.node_id, missing=miss))
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Self-test on the real ES2002 entity trap.
    corpus = {
        "ES2002a.C": "Um I'm Craig and I'm User Interface.",
        "ES2002d.C": "So the marketing side of things, I looked at the trend watchers.",
    }
    m = Model()

    # offsets computed, never hand-guessed -- the assertion is unforgiving
    s = corpus["ES2002a.C"].index("Craig")
    cite_craig = Citation("ES2002a.C", s, s + 5, "Craig")

    craig = Participant(id="participant:1", appears_in=["ES2002a.C", "ES2002d.C"])
    craig.fields["name"] = Extracted("Craig", cite_craig)
    craig.identity_evidence["ES2002a.C"] = cite_craig
    # deliberately NO evidence for ES2002d.C -- this is the real ES2002 trap:
    # the letter C refers to Craig in meeting a and to Andrew in meeting d
    m.add(craig)

    bad = Stakeholder(id="stakeholder:1")
    bad.fields["label"] = Derived("an efficient student", "stakeholder_label",
                                  ["persona:1", "opcontext:1"])
    m.add(bad)

    print("VIOLATIONS")
    vs = check(m, corpus)
    for x in vs:
        print("  ", x)
    print("\nESCALATIONS (what the system says instead of guessing)")
    for q in escalate(vs):
        print(f"   oq {q.gate} -> {q.attach_to}: {q.question}")
