#!/usr/bin/env python3
"""Driver: reproduces every finding from the 21 Sep investigation.

    py run.py C:\\Users\\Mukundh\\ami\\manual
    python3 run.py ~/ami/manual

Needs ami.py in the same folder.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ami")
ROOT = os.path.abspath(os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
os.environ["AMI_ROOT"] = ROOT

import ami  # noqa: E402  (after AMI_ROOT is set)

TRIAL = sys.argv[2] if len(sys.argv) > 2 else "ES2002"
MEETINGS = [f"{TRIAL}{m}" for m in "abcd"]


def rule(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


if not os.path.isdir(ROOT):
    sys.exit(f"not a directory: {ROOT}")
print(f"AMI_ROOT = {ROOT}\ntrial    = {TRIAL}")

loaded = {}
for mid in MEETINGS:
    try:
        loaded[mid] = ami.load_meeting(mid, ROOT)
    except FileNotFoundError as e:
        print(f"  skip {mid}: {e}")
if not loaded:
    sys.exit(f"no meetings loaded for {TRIAL} -- check the path")

# ---------------------------------------------------------------- 1
rule("1. WHAT LOADED")
for mid, m in loaded.items():
    print(f"  {mid}: {len(m.docs)} speakers, {len(m.utterances):>4} utterances, "
          f"{len(m.acts):>4} dialogue acts, {len(m.summlinks):>3} gold citations")

# ---------------------------------------------------------------- 2
rule("2. TURN INTERLEAVING RESTORED  (first 14 utterances of meeting a)")
first = loaded[MEETINGS[0]]
shown = 0
for u in first.utterances:
    if u.text.strip():
        print(f"  [{u.start:7.2f}] {u.speaker}: {u.text[:86]}")
        shown += 1
        if shown >= 14:
            break
print("\n  Note how a role claim can be split across an interruption -- any chunker")
print("  that slices on a fixed window breaks that evidence in half.")

# ---------------------------------------------------------------- 3
rule("3. GOLD CITATIONS  (claim -> supporting spans, resolved and verified)")
m = first
links = {}
for sid, aid in m.summlinks:
    links.setdefault(sid, []).append(aid)
for sid, txt in m.summary.get("decisions", [])[:3]:
    print(f"\n  DECISION: {txt}")
    for aid in links.get(sid, [])[:3]:
        a = m.acts.get(aid)
        if not a or not a.span:
            continue
        print(f"    {a.doc_id}[{a.span[0]}:{a.span[1]}]  spk={a.speaker}"
              f"({m.roles.get(a.speaker,'?')})  type={a.da_type}")
        print(f"      \"{a.text[:92]}\"")

ok = broken = 0
for mid, mm in loaded.items():
    for sid, aid in mm.summlinks:
        a = mm.acts.get(aid)
        if a and a.span and mm.docs[a.speaker].text[a.span[0]:a.span[1]] == a.text:
            ok += 1
        else:
            broken += 1
print(f"\n  CHAIN INTEGRITY across the trial: {ok} resolved, {broken} broken")

# ---------------------------------------------------------------- 4
rule("4. REQUIREMENT EVOLUTION  (proposed -> committed -> superseded)")
for mid in MEETINGS:
    if mid not in loaded:
        continue
    mm = loaded[mid]
    print(f"\n  ---------- {mid} ----------")
    for sec in ("decisions", "problems"):
        for _, txt in mm.summary.get(sec, []):
            print(f"    [{sec:<9}] {txt}")

rule("4b. TRACKING ONE TOPIC ACROSS THE TRIAL")
for kw in ("batter", "LCD", "button", "speech recognition"):
    print(f"\n  == '{kw}' ==")
    for mid in MEETINGS:
        if mid not in loaded:
            continue
        for sec in ("decisions", "problems"):
            for _, txt in loaded[mid].summary.get(sec, []):
                if kw.lower() in txt.lower():
                    print(f"    {mid} [{sec:<9}] {txt}")

# ---------------------------------------------------------------- 5
rule("5. ROLE EXTRACTION FROM TEXT  vs  GOLD METADATA")
PAT = [(r"\bproject manager\b", "PM"), (r"\bmarketing expert\b", "ME"),
       (r"\bmarketing\b", "ME"), (r"\buser interface\b", "UI"),
       (r"\binterface designer\b", "UI"), (r"\bindustrial designer\b", "ID")]
extracted = {}
for spk, doc in sorted(first.docs.items()):
    hit = None
    for u in first.utterances:
        if u.speaker != spk or not u.text:
            continue
        for pat, role in PAT:
            mm2 = re.search(pat, u.text, re.I)
            if mm2 and re.search(r"\bI'?m\b|\bI am\b", u.text, re.I):
                hit = (role, u)
                break
        if hit:
            break
    if hit:
        role, u = hit
        extracted[spk] = role
        print(f"  {spk} -> {role}   cite {u.doc_id}[{u.span[0]}:{u.span[1]}]")
        print(f"       \"{u.text[:88]}\"")
    else:
        print(f"  {spk} -> UNGROUNDED  (no self-stated role in this meeting)")
# This script is a diagnostic, so it may look at gold -- but only through the
# scorer-side door, and only after the extraction above has finished.
gold_roles = ami.load_roles(MEETINGS[0])
print(f"\n  extracted from text : {extracted}")
print(f"  gold metadata       : {dict(sorted(gold_roles.items()))}")
agree = [s for s in extracted if extracted[s] == gold_roles.get(s)]
print(f"  agreement           : {len(agree)}/{len(extracted)}")

# ---------------------------------------------------------------- 6
rule("6. THE ENTITY-RESOLUTION TRAP")
import xml.etree.ElementTree as ET
mp = ami.GOLD_PATH
if not os.path.exists(mp):
    sys.exit(f"gold labels not found at {mp} -- set GOLD_PATH")
rows = {}
for meet in ET.parse(mp).getroot():
    obs = meet.attrib.get("observation", "")
    if obs.startswith(TRIAL):
        rows[obs] = {s.attrib["nxt_agent"]: (s.attrib.get("global_name"),
                                             s.attrib.get("role"))
                     for s in meet if "nxt_agent" in s.attrib}
print(f"  {'meeting':<10}" + "".join(f"{s:>22}" for s in "ABCD"))
for obs in sorted(rows):
    d = rows[obs]
    print(f"  {obs:<10}" + "".join(f"{d[s][0]+'/'+d[s][1]:>22}"
                                   for s in "ABCD" if s in d))

people = {}
for obs in sorted(rows):
    for letter, (gn, role) in rows[obs].items():
        people.setdefault(gn, set()).add(letter)
unstable = {g: ls for g, ls in people.items() if len(ls) > 1}
print("\n  people whose speaker LETTER changes during the trial:")
if unstable:
    for g, ls in unstable.items():
        print(f"    {g} appears as {sorted(ls)}  <-- doc_id is NOT a stable entity key")
    print("\n  Consequence: keying stakeholders on the speaker letter merges two")
    print("  different people. All five gates pass. Every citation is CORRECT.")
    print("  Nothing is fabricated. The model is still wrong.")
else:
    print("    none -- letters are stable in this trial")

print("\n  talk volume follows the person, not the letter:")
for mid in MEETINGS:
    if mid not in loaded:
        continue
    mm = loaded[mid]
    cells = []
    for s in "ABCD":
        if s in mm.docs:
            gn = rows.get(mid, {}).get(s, ("?", "?"))
            cells.append(f"{s}={gn[0]}/{gn[1]}:{len(mm.docs[s].text)}")
    print(f"    {mid}  " + "  ".join(cells))

rule("DONE")
print("  Findings written up in CORPUS_NOTES.md")
