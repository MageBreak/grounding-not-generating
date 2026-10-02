#!/usr/bin/env python3
"""The fabrication experiment: grounded extraction vs a naive baseline.

    py prototype.py C:\\Users\\Mukundh\\ami\\manual
    python3 prototype.py ~/ami/manual

Two extractors run over the kickoff meeting of every AMI scenario trial.

  grounded : asserts a role only where a speaker states it about themselves,
             and attaches a verifiable character span. Where the evidence is
             absent it emits an OpenQuestion instead of a guess.

  naive    : always produces a complete answer. Where evidence is missing it
             falls back on the scenario prior (PM/ME/UI/ID by speaker order) --
             which is exactly what a language model does under gate pressure.

Both are scored against corpusResources/meetings.xml, which is GOLD LABELS and
which NEITHER EXTRACTOR MAY READ. Only the scorer opens it.
"""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ami")
ROOT = os.path.abspath(os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
os.environ["AMI_ROOT"] = ROOT

import ami            # noqa: E402
from schema import Citation, OpenQuestion   # noqa: E402

ROLE_PATTERNS = [
    (r"\bproject manager\b", "PM"),
    (r"\bmarketing expert\b", "ME"),
    (r"\buser interface\b", "UI"),
    (r"\binterface design", "UI"),
    (r"\bindustrial design", "ID"),
    (r"\bmarketing\b", "ME"),
]
SELF_REF = re.compile(r"\b(I'?m|I am|my role is|my name is)\b", re.I)
# Second person: the speaker is assigning a role to SOMEONE ELSE.
# IS1000a.D -- "You are the marketing guy? ... So you are the marketing."
OTHER_REF = re.compile(r"\b(you'?re|you are|your role|he'?s|she'?s|they'?re)\b", re.I)
SCENARIO_PRIOR = ["PM", "ME", "UI", "ID"]      # the conventional ordering

SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
MAX_SPAN = 240          # a self-introduction does not need more than this


# --------------------------------------------------------------------------- #
def _sentences(text: str):
    """Yield (offset_within_text, sentence)."""
    pos = 0
    for s in SENT_SPLIT.split(text):
        i = text.find(s, pos)
        if i < 0:
            i = pos
        yield i, s
        pos = i + len(s)


def extract_grounded(meeting) -> tuple[dict, list[OpenQuestion], dict]:
    """Assert a role only where a citable span supports it.

    Three rules learned from the errors this extractor made on 21 Sep:

      1. cite the SENTENCE, not the whole utterance -- a 759-character window
         catches a stray "I'm" near a stray role word (TS3011a)
      2. take the role term that appears EARLIEST IN THE TEXT, not the first
         one in the pattern list -- "my role is industrial designer ... deal
         with the technical-functional designs" was read as UI (IS1009a)
      3. reject second-person attributions -- "you are the marketing" is a
         claim about someone else (IS1000a, and ES2003's roll call)
    """
    roles, cites = {}, {}
    for u in meeting.utterances:
        if u.speaker in roles or not u.text or not u.span:
            continue
        for off, sent in _sentences(u.text):
            if not SELF_REF.search(sent) or OTHER_REF.search(sent):
                continue
            # earliest role mention wins
            best = None
            for pat, role in ROLE_PATTERNS:
                m = re.search(pat, sent, re.I)
                if m and (best is None or m.start() < best[0]):
                    best = (m.start(), role)
            if not best:
                continue
            s = u.span[0] + off
            e = s + len(sent)
            if e - s > MAX_SPAN:          # refuse to cite a loose window
                continue
            roles[u.speaker] = best[1]
            cites[u.speaker] = Citation(u.doc_id, s, e,
                                        meeting.docs[u.speaker].text[s:e])
            break

    questions = [
        OpenQuestion(
            id=f"oq:{meeting.meeting_id}:{spk}",
            question=f"What is the role of speaker {spk} in {meeting.meeting_id}?",
            gate="G5", attach_to=f"participant:{meeting.meeting_id}.{spk}",
            missing="role", resolution="C2_ask_human",
        )
        for spk in sorted(meeting.docs) if spk not in roles
    ]
    return roles, questions, cites


def extract_naive(meeting, grounded: dict) -> dict:
    """Always complete. Fills every gap from the scenario prior."""
    out = {}
    for i, spk in enumerate(sorted(meeting.docs)):
        out[spk] = grounded.get(spk) or SCENARIO_PRIOR[i % len(SCENARIO_PRIOR)]
    return out


def load_gold(root: str) -> dict[str, dict[str, str]]:
    """GOLD LABELS. Scorer only -- never the extractors."""
    p = os.path.join(root, "corpusResources", "meetings.xml")
    out = {}
    for m in ET.parse(p).getroot():
        obs = m.attrib.get("observation", "")
        out[obs] = {s.attrib["nxt_agent"]: s.attrib.get("role")
                    for s in m if "nxt_agent" in s.attrib}
    return out


# --------------------------------------------------------------------------- #
def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"not a directory: {ROOT}")
    gold = load_gold(ROOT)

    wordsdir = os.path.join(ROOT, "words")
    trials = sorted({f[:6] for f in os.listdir(wordsdir) if f[:2] in ("ES", "IS", "TS")})

    print(f"AMI_ROOT = {ROOT}")
    print(f"trials   = {len(trials)} scenario trials, kickoff meeting of each\n")
    print(f"{'trial':<9}{'cited':>6}{'ok':>4}   {'naive':>6}{'ok':>4}   "
          f"{'fabricated':>11}{'wrong':>7}")
    print("-" * 56)

    G = GOK = N = NOK = F = FW = 0
    fab_examples, grounded_errors = [], []

    for t in trials:
        mid = f"{t}a"
        if mid not in gold:
            continue
        try:
            m = ami.load_meeting(mid, ROOT)
        except Exception:
            continue
        if not m.docs:
            continue

        gr, qs, cites = extract_grounded(m)
        nv = extract_naive(m, gr)
        g = gold[mid]

        gok = sum(1 for s, v in gr.items() if g.get(s) == v)
        nok = sum(1 for s, v in nv.items() if g.get(s) == v)
        fab = [s for s in nv if s not in gr]
        fw = [s for s in fab if nv[s] != g.get(s)]

        G += len(gr); GOK += gok; N += len(nv); NOK += nok
        F += len(fab); FW += len(fw)

        for s, v in gr.items():
            if g.get(s) != v:
                grounded_errors.append((mid, s, v, g.get(s), cites[s]))
        if fw and len(fab_examples) < 3:
            s = fw[0]
            fab_examples.append((mid, s, nv[s], g.get(s)))

        print(f"{t:<9}{len(gr):>6}{gok:>4}   {len(nv):>6}{nok:>4}   "
              f"{len(fab):>11}{len(fw):>7}")

    print("-" * 56)
    print(f"""
RESULTS  ({N} speaker-role assertions)

  grounded extractor
     asserted   {G:>4}      precision {GOK / max(G, 1):.3f}      coverage {G / max(N, 1):.3f}
     correct    {GOK:>4}      refused to answer {N - G} times

  naive extractor  (always complete, always passes all five gates)
     asserted   {N:>4}      precision {NOK / max(N, 1):.3f}      coverage 1.000
     correct    {NOK:>4}

  FABRICATION RATE = {F}/{N} = {F / max(N, 1):.3f}
     of those {F} unevidenced claims, {FW} were WRONG ({FW / max(F, 1):.1%})

The naive extractor looks better by every structural measure: complete,
confident, all five gates green. It is wrong {1 - NOK / max(N, 1):.0%} of the time.
The grounded extractor answers less often and is right {GOK / max(G, 1):.0%} of the time
when it does.""")

    if fab_examples:
        print("\n  fabrications (no evidence, and wrong):")
        for mid, s, got, want in fab_examples:
            print(f"     {mid} speaker {s}: claimed {got}, actually {want}")

    if grounded_errors:
        print("\n  GROUNDED EXTRACTOR ERRORS -- string containment passed,")
        print("  but the span does not support the claim. This is what")
        print("  grounding validation (layer 2) exists to catch:")
        for mid, s, got, want, c in grounded_errors:
            print(f"\n     {mid} speaker {s}: extracted {got}, gold {want}")
            print(f"     cite {c}")
            print(f"     \"{c.exact_text[:150]}\"")


if __name__ == "__main__":
    main()
