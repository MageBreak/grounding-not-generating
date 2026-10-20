"""The ontology layer: what our types *are*, and the gates that follow.

schema.py lists eight gates. This module does not list them -- it declares what
each node type fundamentally is, using UFO/OntoUML stereotypes, and derives the
constraints from the stereotype.

The difference matters. "Every Need must have a holder" is a rule somebody
wrote down and could have forgotten. "Need is a Mode, and a Mode is an aspect
that inheres in a bearer" is a statement about what a Need is; the constraint
is a consequence, and a Need without a bearer is not an incomplete record but
an impossible object.

Three payoffs, in order of how much they are worth:

  1. Existing gates become derivable. G1, G2, G4 and G8 all fall out of four
     stereotype assignments. `explain_gates()` prints the derivation.
  2. New constraints appear for free. A Role with no identity provider is
     invalid whether or not anything points at it -- schema.py misses this.
  3. The declaration is a map of the model, including the parts not built yet.
     Types marked implemented=False are honest gaps, not silent ones.

Standard library only.  Self-test:  python3 ontology.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from schema import Model, Node, Violation

# --------------------------------------------------------------------------- #
# Stereotypes (UFO / OntoUML, plus gUFO's information artifact)
# --------------------------------------------------------------------------- #


class Stereotype(Enum):
    """Each carries a metaphysical commitment, spelled out in STEREOTYPE_DOC."""
    KIND = "kind"
    SUBKIND = "subkind"
    ROLE = "role"
    PHASE = "phase"
    RELATOR = "relator"
    MODE = "mode"
    QUALITY = "quality"
    CATEGORY = "category"
    SITUATION = "situation"
    EVENT = "event"
    INFORMATION_ARTIFACT = "information_artifact"


STEREOTYPE_DOC: dict[Stereotype, str] = {
    Stereotype.KIND:
        "Rigid and identity-supplying. A thing cannot stop being one without "
        "ceasing to exist. Everything else borrows its identity from a Kind.",
    Stereotype.SUBKIND:
        "Rigid specialisation of a Kind. Inherits identity, adds no new one.",
    Stereotype.ROLE:
        "Anti-rigid and relationally dependent. A thing plays it only while "
        "standing in some relation, and stops without ceasing to exist.",
    Stereotype.PHASE:
        "Anti-rigid and intrinsically dependent. Holds while the bearer's own "
        "properties hold. Phases of a type partition it.",
    Stereotype.RELATOR:
        "The truth-maker of a material relation. Exists only by mediating two "
        "or more relata; it is what makes the relation hold.",
    Stereotype.MODE:
        "An aspect that inheres in a single bearer and cannot float free. "
        "Existentially dependent, not relational.",
    Stereotype.QUALITY:
        "An inherent aspect measurable in some value space.",
    Stereotype.CATEGORY:
        "Rigid but non-sortal: groups things of several Kinds, supplies no "
        "identity of its own.",
    Stereotype.SITUATION:
        "A configuration of entities obtaining at a time. Not a thing; a state "
        "of affairs that other things can be relative to.",
    Stereotype.EVENT:
        "A perduring occurrence with temporal parts. Happens rather than exists.",
    Stereotype.INFORMATION_ARTIFACT:
        "A created object that is *about* something. Its subject is part of "
        "what it is; an artifact about nothing is not an artifact.",
}

#: Stereotypes whose instances can stop being that type without ceasing to be.
ANTI_RIGID = {Stereotype.ROLE, Stereotype.PHASE}

#: Stereotypes that supply identity to their instances.
IDENTITY_SUPPLYING = {Stereotype.KIND, Stereotype.SUBKIND}


# --------------------------------------------------------------------------- #
# Type declarations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TypeDecl:
    name: str
    stereotype: Stereotype
    node_prefix: str | None = None      # the id prefix used in schema.py
    implemented: bool = True            # False == a declared gap, not a silent one

    identity_from: str | None = None    # Roles/Phases: the Kind behind them
    depends_on: tuple[str, ...] = ()    # Roles/artifacts: relational dependencies
    inheres_in: str | None = None       # Modes/Qualities: the single bearer
    mediates: tuple[str, ...] = ()      # Relators: the relata
    about: str | None = None            # InformationArtifact: its subject

    #: dependency type name -> attribute on the schema.py node that realises it
    bindings: dict[str, str] = field(default_factory=dict)

    #: Phase partition carried as a field on this type, if any.
    phases: tuple[str, ...] = ()
    phase_attr: str | None = None
    phase_evidence_attr: str | None = None
    #: phases whose entry is a change of state and so needs its own evidence
    phases_needing_evidence: tuple[str, ...] = ()

    note: str = ""


S = Stereotype

ONTOLOGY: dict[str, TypeDecl] = {
    # ---- identity-supplying Kinds -----------------------------------------
    "Participant": TypeDecl(
        "Participant", S.KIND, node_prefix="participant",
        note="A real person. NOT a speaker channel. Channel letters are "
             "document-local labels; the person is what persists, which is "
             "exactly why cross-document identity needs its own evidence.",
    ),
    "Persona": TypeDecl(
        "Persona", S.KIND, node_prefix="persona", implemented=False,
        note="An abstracted archetype of a person, not a person.",
    ),
    "Product": TypeDecl(
        "Product", S.KIND, node_prefix="product", implemented=False,
    ),

    # ---- the situation a persona is in ------------------------------------
    "OperatingContext": TypeDecl(
        "OperatingContext", S.SITUATION, node_prefix="opcontext", implemented=False,
        note="A state of affairs, not an entity. A persona is the same persona "
             "in two contexts; the stakeholder role differs.",
    ),

    # ---- the Role, which is where most of the trouble lives ----------------
    "Stakeholder": TypeDecl(
        "Stakeholder", S.ROLE, node_prefix="stakeholder",
        identity_from="Persona",
        depends_on=("OperatingContext",),
        bindings={"Persona": "persona_ref", "OperatingContext": "context_ref"},
        note="Nobody is a stakeholder simpliciter. You are a stakeholder of "
             "something, in some context, and you stop being one without "
             "ceasing to exist. Anti-rigid + relationally dependent.",
    ),

    # ---- the Mode ----------------------------------------------------------
    "Need": TypeDecl(
        "Need", S.MODE, node_prefix="need",
        inheres_in="Stakeholder",
        bindings={"Stakeholder": "held_by"},
        note="A need is an aspect of someone. A need with no holder is like a "
             "colour with no object -- not missing data, not a possible thing.",
    ),

    # ---- the artifact ------------------------------------------------------
    "Requirement": TypeDecl(
        "Requirement", S.INFORMATION_ARTIFACT, node_prefix="req",
        about="Need",
        depends_on=("Participant",),
        bindings={"Need": "serves", "Participant": "raised_by"},
        phases=("proposed", "committed", "superseded", "rejected"),
        phase_attr="status",
        phase_evidence_attr="status_citation",
        phases_needing_evidence=("committed", "superseded", "rejected"),
        note="A requirement is a document-like object about a need, authored "
             "by someone. Its status is a Phase: entering a non-default phase "
             "is an event in the world and needs its own evidence.",
    ),

    # ---- events, declared but not built ------------------------------------
    "Journey": TypeDecl(
        "Journey", S.EVENT, node_prefix="journey", implemented=False,
        depends_on=("Stakeholder",),
        bindings={"Stakeholder": "travelled_by"},
    ),
    "JourneyStep": TypeDecl(
        "JourneyStep", S.EVENT, node_prefix="step", implemented=False,
        depends_on=("Journey",),
        bindings={"Journey": "step_of"},
        note="A temporal part of a Journey. Mereology, not reference: a step "
             "is part of the journey, it does not merely point at it.",
    ),
}


def by_prefix() -> dict[str, TypeDecl]:
    return {d.node_prefix: d for d in ONTOLOGY.values() if d.node_prefix}


# --------------------------------------------------------------------------- #
# Meta-check: are the declarations themselves coherent?
# --------------------------------------------------------------------------- #


def validate_ontology() -> list[str]:
    """Check the ontology before trusting anything it says about data."""
    problems: list[str] = []
    for d in ONTOLOGY.values():
        if d.stereotype in ANTI_RIGID and not d.identity_from and not d.phases:
            problems.append(
                f"{d.name}: anti-rigid ({d.stereotype.value}) but names no "
                f"identity_from -- an anti-rigid type cannot supply identity, "
                f"so something else must")
        if d.identity_from:
            src = ONTOLOGY.get(d.identity_from)
            if src is None:
                problems.append(f"{d.name}: identity_from {d.identity_from!r} is undeclared")
            elif src.stereotype not in IDENTITY_SUPPLYING:
                problems.append(
                    f"{d.name}: identity_from {src.name} is a "
                    f"{src.stereotype.value}, which supplies no identity")
        if d.stereotype is S.MODE and not d.inheres_in:
            problems.append(f"{d.name}: a Mode must name the bearer it inheres in")
        if d.stereotype is S.RELATOR and len(d.mediates) < 2:
            problems.append(f"{d.name}: a Relator must mediate at least two relata")
        if d.stereotype is S.INFORMATION_ARTIFACT and not d.about:
            problems.append(f"{d.name}: an information artifact must be about something")
        for dep in (*d.depends_on, *(x for x in (d.inheres_in, d.about) if x)):
            if dep not in ONTOLOGY:
                problems.append(f"{d.name}: depends on undeclared type {dep!r}")
            elif dep not in d.bindings:
                problems.append(f"{d.name}: dependency on {dep} has no binding "
                                f"-- name the attribute that realises it")
    return problems


# --------------------------------------------------------------------------- #
# Derived constraints -- these ARE the gates, obtained rather than listed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DerivedConstraint:
    code: str                 # ontology gate id
    equivalent_gate: str      # the hand-written gate in schema.py, if any
    type_name: str
    statement: str
    because: str


def derive_constraints() -> list[DerivedConstraint]:
    out: list[DerivedConstraint] = []
    for d in ONTOLOGY.values():
        if not d.implemented:
            continue

        if d.stereotype is S.MODE and d.inheres_in:
            out.append(DerivedConstraint(
                "O-MODE-BEARER", "G4", d.name,
                f"every {d.name} must name a {d.inheres_in} "
                f"(via .{d.bindings[d.inheres_in]})",
                "a Mode is an aspect of a bearer; without one there is nothing "
                "for it to be an aspect of"))

        if d.stereotype is S.ROLE:
            if d.identity_from:
                out.append(DerivedConstraint(
                    "O-ROLE-IDENTITY", "-", d.name,
                    f"every {d.name} must name a {d.identity_from} "
                    f"(via .{d.bindings[d.identity_from]})",
                    "a Role is anti-rigid and supplies no identity of its own; "
                    "some Kind must say who is playing it"))
            for dep in d.depends_on:
                out.append(DerivedConstraint(
                    "O-ROLE-RELATUM", "G4", d.name,
                    f"every {d.name} must name a {dep} (via .{d.bindings[dep]})",
                    "a Role is relationally dependent; the relation it depends "
                    "on must actually be there"))

        if d.stereotype is S.INFORMATION_ARTIFACT:
            if d.about:
                out.append(DerivedConstraint(
                    "O-ARTIFACT-SUBJECT", "G2", d.name,
                    f"every {d.name} must be about at least one {d.about} "
                    f"(via .{d.bindings[d.about]})",
                    "an information artifact is individuated by its subject; "
                    "one about nothing is not one"))
            for dep in d.depends_on:
                out.append(DerivedConstraint(
                    "O-ARTIFACT-AUTHOR", "G1", d.name,
                    f"every {d.name} must name a {dep} (via .{d.bindings[dep]})",
                    "artifacts are created by someone; authorship is part of "
                    "what the artifact is, not metadata about it"))

        if d.phases and d.phase_attr:
            out.append(DerivedConstraint(
                "O-PHASE-PARTITION", "-", d.name,
                f".{d.phase_attr} must be one of {d.phases}",
                "phases of a type partition it: exactly one holds at a time"))
            if d.phases_needing_evidence and d.phase_evidence_attr:
                out.append(DerivedConstraint(
                    "O-PHASE-TRANSITION", "G8", d.name,
                    f"entering {d.phases_needing_evidence} requires "
                    f".{d.phase_evidence_attr}",
                    "a phase change is an event in the world; the evidence for "
                    "the thing is not evidence for the change"))

        if d.stereotype is S.RELATOR:
            out.append(DerivedConstraint(
                "O-RELATOR-ARITY", "-", d.name,
                f"every {d.name} must mediate {d.mediates}",
                "a Relator exists only in mediating; an unmediating one is not "
                "underspecified, it is nothing"))
    return out


def explain_gates() -> str:
    """Show which hand-written gates fall out of the stereotype assignments."""
    lines = ["Gate                      derived from", "-" * 64]
    seen: dict[str, list[DerivedConstraint]] = {}
    for c in derive_constraints():
        seen.setdefault(c.equivalent_gate, []).append(c)
    for gate in sorted(seen, key=lambda g: (g == "-", g)):
        label = gate if gate != "-" else "(new)"
        for i, c in enumerate(seen[gate]):
            lines.append(f"{label if i == 0 else '':<25} {c.code} on {c.type_name}")
            lines.append(f"{'':<25}   {c.because}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Running the derived constraints over a real model
# --------------------------------------------------------------------------- #


def _ref_values(node: Node, attr: str) -> list[str]:
    v = getattr(node, attr, None)
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def check_ontology(model: Model) -> list[Violation]:
    """Deterministic, like schema.check(). No model call, no judgement.

    Reported alongside schema.check() rather than replacing it: these catch
    impossible objects, those catch ungrounded ones. Different failures.
    """
    decls = by_prefix()
    out: list[Violation] = []

    for node in model.nodes.values():
        d = decls.get(node.kind)
        if d is None:
            out.append(Violation("O-UNDECLARED", node.id,
                                 f"node kind {node.kind!r} has no ontology "
                                 f"declaration -- it is not a known kind of thing"))
            continue

        for c in derive_constraints():
            if c.type_name != d.name:
                continue

            if c.code in ("O-MODE-BEARER", "O-ROLE-IDENTITY",
                          "O-ROLE-RELATUM", "O-ARTIFACT-SUBJECT",
                          "O-ARTIFACT-AUTHOR"):
                attr = c.statement.rsplit(".", 1)[-1].rstrip(")")
                refs = _ref_values(node, attr)
                if not refs:
                    out.append(Violation(c.code, node.id,
                                         f"{c.statement} -- because {c.because}"))
                for r in refs:
                    if r not in model.nodes:
                        out.append(Violation(c.code, node.id,
                                             f".{attr} -> {r} which does not exist"))

            elif c.code == "O-PHASE-PARTITION" and d.phase_attr:
                val = getattr(node, d.phase_attr, None)
                if val is not None and val not in d.phases:
                    out.append(Violation(c.code, node.id,
                                         f".{d.phase_attr}={val!r} is outside "
                                         f"the partition {d.phases}"))

            elif c.code == "O-PHASE-TRANSITION" and d.phase_attr:
                val = getattr(node, d.phase_attr, None)
                if val in d.phases_needing_evidence:
                    if getattr(node, d.phase_evidence_attr, None) is None:
                        out.append(Violation(c.code, node.id,
                                             f"phase {val!r} with no "
                                             f".{d.phase_evidence_attr} -- "
                                             f"{c.because}"))
    return out


def coverage_gaps() -> list[str]:
    """Declared but unimplemented types. The roadmap, stated by the ontology."""
    return [f"{d.name} ({d.stereotype.value})"
            for d in ONTOLOGY.values() if not d.implemented]


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from schema import Citation, Derived, Extracted, Participant, Stakeholder, check

    print("=" * 68)
    print("ONTOLOGY META-CHECK")
    print("=" * 68)
    problems = validate_ontology()
    print("  coherent" if not problems else "\n".join(f"  ! {p}" for p in problems))

    print()
    print("=" * 68)
    print("GATES, DERIVED RATHER THAN LISTED")
    print("=" * 68)
    print(explain_gates())

    print()
    print("=" * 68)
    print("DECLARED BUT NOT BUILT")
    print("=" * 68)
    for g in coverage_gaps():
        print("  ", g)

    print()
    print("=" * 68)
    print("ON THE ES2002 TRAP -- what the ontology catches that the gates do not")
    print("=" * 68)
    corpus = {"ES2002a.C": "Um I'm Craig and I'm User Interface."}
    m = Model()
    s = corpus["ES2002a.C"].index("Craig")
    cite = Citation("ES2002a.C", s, s + 5, "Craig")
    craig = Participant(id="participant:1", appears_in=["ES2002a.C", "ES2002d.C"])
    craig.fields["name"] = Extracted("Craig", cite)
    craig.identity_evidence["ES2002a.C"] = cite
    m.add(craig)
    bad = Stakeholder(id="stakeholder:1")
    bad.fields["label"] = Derived("an efficient student", "stakeholder_label",
                                  ["persona:1", "opcontext:1"])
    m.add(bad)

    g = check(m, corpus)
    o = check_ontology(m)
    print("  schema.check()    ->", len(g), "violations")
    for x in g:
        print("     ", x)
    print("  check_ontology()  ->", len(o), "violations")
    for x in o:
        print("     ", x)
    print()
    print("  The stakeholder is ungrounded (G5/G6) AND impossible (O-ROLE-*):")
    print("  a Role with nobody playing it and no context to play it in is not")
    print("  an incomplete record. Supplying a citation would not fix it.")
