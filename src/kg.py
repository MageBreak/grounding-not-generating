"""The knowledge graph: edges with mandatory provenance, and path equations.

The central design decision is a refusal: there is no `add(s, p, o)`. Every
edge arrives either grounded (carrying a Citation that round-trips) or derived
(carrying a rule and the edges it came from). You cannot assert a relationship
into this graph without saying where it came from.

That is not fussiness. A model with flawless per-node citations can still
invent the *relations* between those nodes, and a node-level citation check
cannot see it. Grounding has to cover the arrows, not just the dots.

The second half of the file is the path-equation checker. Existence gates ask
"is the thing there?". Path equations ask "do two routes through the graph
agree?" -- a different failure class, and the one that survives a model whose
every individual node is correct.

Standard library only. networkx and rdflib are optional, for export.

Self-test:  python3 kg.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from schema import Citation, Model, Violation

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Derivation:
    """Why an edge exists when nobody stated it.

    Mirrors schema.Derived: carries no evidential weight, and -- this is the
    load-bearing part -- may not be used to ground a further edge.
    """
    rule: str
    frm: tuple[str, ...] = ()

    def __repr__(self):
        return f"derived({self.rule})"


Provenance = Citation | Derivation


@dataclass(frozen=True)
class Edge:
    s: str
    p: str
    o: str
    provenance: Provenance

    @property
    def grounded(self) -> bool:
        return isinstance(self.provenance, Citation)

    def key(self) -> tuple[str, str, str]:
        return (self.s, self.p, self.o)

    def __repr__(self):
        return f"({self.s} -{self.p}-> {self.o}) [{self.provenance!r}]"


class ProvenanceRequired(ValueError):
    """Raised on any attempt to add an edge without saying where it came from."""


def edge_ref(s: str, p: str, o: str) -> str:
    """Stable textual handle for an edge, used in Derivation.frm."""
    return f"{s}|{p}|{o}"


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #


class KG:
    def __init__(self) -> None:
        self._edges: list[Edge] = []
        self._out: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._in: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.entities: set[str] = set()
        self.predicates: set[str] = set()

    # -- construction -------------------------------------------------------

    def add_grounded(self, s: str, p: str, o: str, citation: Citation,
                     corpus: dict[str, str] | None = None) -> Edge:
        """Add an edge stated by a passage. The citation must round-trip."""
        if not isinstance(citation, Citation):
            raise ProvenanceRequired("add_grounded needs a Citation")
        if corpus is not None and not citation.verify(corpus):
            raise ValueError(f"citation does not round-trip: {citation}")
        return self._insert(Edge(s, p, o, citation))

    def add_derived(self, s: str, p: str, o: str, rule: str,
                    frm: Iterable[str] = ()) -> Edge:
        """Add an edge computed from other edges. Never grounds anything else."""
        if not rule:
            raise ProvenanceRequired("add_derived needs a rule")
        return self._insert(Edge(s, p, o, Derivation(rule, tuple(frm))))

    def _insert(self, e: Edge) -> Edge:
        # The invariant, enforced at the single point every edge passes through,
        # so a direct _insert cannot smuggle an unprovenanced edge in either.
        if not isinstance(e.provenance, (Citation, Derivation)):
            raise ProvenanceRequired(
                f"edge {e.s} -{e.p}-> {e.o} has no provenance; use "
                f"add_grounded() or add_derived()")
        self._edges.append(e)
        self._out[(e.s, e.p)].add(e.o)
        self._in[(e.o, e.p)].add(e.s)
        self.entities.update((e.s, e.o))
        self.predicates.add(e.p)
        return e

    # -- reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._edges)

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._edges)

    def grounded(self) -> list[Edge]:
        return [e for e in self._edges if e.grounded]

    def derived(self) -> list[Edge]:
        return [e for e in self._edges if not e.grounded]

    def out(self, node: str, pred: str) -> set[str]:
        return set(self._out.get((node, pred), ()))

    def follow(self, node: str, path: Iterable[str]) -> set[str]:
        """Walk a chain of predicates. Returns every node reachable by that path.

        Sets, not single values, because most of these relations are many-to-
        many. Composition of relations is still composition.
        """
        cur = {node}
        for pred in path:
            nxt: set[str] = set()
            for n in cur:
                nxt |= self.out(n, pred)
            cur = nxt
            if not cur:
                break
        return cur

    def ungrounded_relations(self) -> list[Violation]:
        """Derived edges whose inputs are themselves derived.

        Derivation standing on derivation: nothing underneath it ever touched
        the corpus. This is schema.py's G6 lifted from nodes to edges, and it
        is the reason `frm` holds edge refs rather than bare node ids.
        """
        derived_refs = {edge_ref(*e.key()) for e in self.derived()}
        out: list[Violation] = []
        for e in self.derived():
            assert isinstance(e.provenance, Derivation)
            culprits = [r for r in e.provenance.frm if r in derived_refs]
            if culprits:
                out.append(Violation(
                    "G6-edge", edge_ref(*e.key()),
                    f"derived from other derived edges only: {culprits}"))
        return out

    # -- export -------------------------------------------------------------

    def to_networkx(self):
        import networkx as nx
        g = nx.MultiDiGraph()
        for e in self._edges:
            g.add_edge(e.s, e.o, key=e.p, predicate=e.p,
                       grounded=e.grounded, provenance=repr(e.provenance))
        return g

    def to_nquads(self) -> str:
        """N-Quads, one named graph per edge, named by its provenance.

        Named graphs are exactly the right tool here: the quad's fourth term
        says *on whose authority* the triple holds. A SPARQL query can then ask
        for grounded edges only, which is the query that matters.
        """
        def uri(x: str) -> str:
            return f"<urn:gng:{x.replace(' ', '_')}>"

        lines = []
        for i, e in enumerate(self._edges):
            if isinstance(e.provenance, Citation):
                c = e.provenance
                g = f"<urn:gng:cite:{c.doc_id}:{c.start}:{c.end}>"
            else:
                g = f"<urn:gng:derived:{i}:{e.provenance.rule}>"
            lines.append(f"{uri(e.s)} {uri(e.p)} {uri(e.o)} {g} .")
        return "\n".join(lines)

    def to_pykeen_triples(self, grounded_only: bool = True) -> list[tuple[str, str, str]]:
        """Triples for embedding. Grounded only, by default and on purpose.

        Training on derived edges teaches the embedding the shape of our own
        inference rules and then lets it confirm them. Grounded edges are the
        only ones that came from outside.
        """
        src = self.grounded() if grounded_only else self._edges
        return [(e.s, e.p, e.o) for e in src]


# --------------------------------------------------------------------------- #
# Building a KG from a validated product model
# --------------------------------------------------------------------------- #

#: schema.py node attribute -> predicate name. Attributes holding references
#: become edges; everything else stays a field on the node.
REFERENCE_ATTRS: dict[str, str] = {
    "raised_by": "raised_by",
    "serves": "serves",
    "held_by": "held_by",
    "persona_ref": "persona_of",
    "context_ref": "in_context",
    "supersedes": "supersedes",
    "step_of": "step_of",
    "travelled_by": "travelled_by",
}


def from_model(model: Model, corpus: dict[str, str] | None = None) -> KG:
    """Project a schema.Model into a KG, carrying provenance across.

    An edge is grounded when the node at its tail has a citation for the field
    that states the reference. Where a reference exists with no field-level
    citation behind it, the edge is recorded as derived -- visibly, rather than
    being quietly promoted to fact.
    """
    kg = KG()
    for node in model.nodes.values():
        for attr, pred in REFERENCE_ATTRS.items():
            v = getattr(node, attr, None)
            if v is None:
                continue
            targets = v if isinstance(v, (list, tuple, set)) else [v]
            ex = node.extracted()
            cite = ex[attr].citation if attr in ex else None
            for t in targets:
                if cite is not None:
                    kg.add_grounded(node.id, pred, t, cite, corpus)
                else:
                    kg.add_derived(node.id, pred, t,
                                   rule=f"reference_attr:{attr}", frm=[node.id])
    return kg


# --------------------------------------------------------------------------- #
# Path equations -- the consistency gates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PathEquation:
    code: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    name: str
    why: str


PATH_EQUATIONS: list[PathEquation] = [
    PathEquation(
        "P1",
        left=("serves", "held_by"),
        right=("raised_by",),
        name="requirement / stakeholder agreement",
        why="the stakeholder a requirement reaches through the need it serves "
            "should not contradict the person recorded as raising it",
    ),
    PathEquation(
        "P2",
        left=("supersedes", "serves"),
        right=("serves",),
        name="supersession confluence",
        why="a requirement that supersedes another should serve a need the "
            "superseded one served; otherwise it replaced something unrelated",
    ),
    PathEquation(
        "P3",
        left=("held_by", "in_context"),
        right=("raised_by",),
        name="need / context agreement",
        why="the context a need reaches through its holder should be one the "
            "raising participant was actually in",
    ),
]


def check_paths(kg: KG, equations: Iterable[PathEquation] = ()) -> list[Violation]:
    """Flag nodes where two routes disagree.

    Semantics worth being careful about: because these relations are many-to-
    many, demanding set *equality* would fire constantly on perfectly good
    models. What is a genuine contradiction is **disjointness** -- both routes
    lead somewhere, and nowhere they lead is shared. A route that leads nowhere
    is silent: that is an existence problem, and the existence gates own it.
    """
    eqs = list(equations) or PATH_EQUATIONS
    out: list[Violation] = []
    for eq in eqs:
        for node in sorted(kg.entities):
            a = kg.follow(node, eq.left)
            b = kg.follow(node, eq.right)
            if a and b and not (a & b):
                out.append(Violation(
                    eq.code, node,
                    f"{eq.name}: {'.'.join(eq.left)} -> {sorted(a)} but "
                    f"{'.'.join(eq.right)} -> {sorted(b)}; disjoint. {eq.why}"))
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    corpus = {
        "m1": "Craig said the remote needs a bigger power button. "
              "Andrew raised the cost ceiling.",
    }

    def cite(text: str) -> Citation:
        i = corpus["m1"].index(text)
        return Citation("m1", i, i + len(text), text)

    kg = KG()

    print("=" * 68)
    print("PROVENANCE IS NOT OPTIONAL")
    print("=" * 68)
    try:
        kg._insert(Edge("req:1", "serves", "need:1", None))   # type: ignore[arg-type]
        print("  !! an edge with no provenance was accepted -- BUG")
    except ProvenanceRequired as e:
        print(f"  even a direct _insert is refused: {e}")
    print("  there is no add(s, p, o). The two constructors are:")
    print("    add_grounded(s, p, o, citation)   -- citation must round-trip")
    print("    add_derived (s, p, o, rule, frm)  -- and may never ground a child")
    try:
        kg.add_grounded("req:1", "serves", "need:1",
                        Citation("m1", 0, 5, "WRONG"), corpus)
    except ValueError as e:
        print(f"  rejected a citation that does not round-trip: {e}")

    print()
    print("=" * 68)
    print("A MODEL WHERE EVERY NODE IS PERFECT AND THE GRAPH IS NOT")
    print("=" * 68)
    kg = KG()
    c1 = cite("Craig said the remote needs a bigger power button")
    c2 = cite("Andrew raised the cost ceiling")
    kg.add_grounded("req:1", "serves", "need:1", c1, corpus)
    kg.add_grounded("need:1", "held_by", "stakeholder:craig", c1, corpus)
    kg.add_grounded("req:1", "raised_by", "stakeholder:andrew", c2, corpus)

    print(f"  {len(kg)} edges, all grounded, every citation round-trips.")
    for e in kg:
        print("   ", e)
    print()
    print("  schema-style existence gates: nothing missing, all green.")
    print("  path equations:")
    vs = check_paths(kg)
    for v in vs:
        print("   ", v)
    if not vs:
        print("    (none)")

    print()
    print("=" * 68)
    print("EXPORT")
    print("=" * 68)
    print(kg.to_nquads())
    print()
    print("  pykeen triples (grounded only):", kg.to_pykeen_triples())
