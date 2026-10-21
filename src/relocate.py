#!/usr/bin/env python3
"""Split "fabricated" into two things it was never one thing.

`locate()` asks whether a quote appears in the document byte for byte. When it
does not, we have been calling that fabrication. ES2012 shows that is wrong:

    model (p0_complete):
      "Right welcome to the first meeting of Real Reaction's development..."
    transcript:
      "Right w welcome to the the first meeting of uh Real Reaction's uh um..."

The model had found the right passage. It tidied the transcription -- dropped
a false start, a stutter, two filler words -- because that is what models are
trained to do with messy text. Under `p1_cite`, which forbids tidying, the
same model on the same trial reproduced every disfluency and all four quotes
located exactly.

So an unlocated quote is one of two very different things:

    PARAPHRASE   the right passage, cleaned up. The model read the document.
                 Our instruction, or our matcher, is at fault.
    FABRICATED   no passage in the document resembles this. The model supplied
                 it from somewhere else.

Only the second belongs in a fabrication rate. Reporting them pooled inflates
the headline number of a project whose entire argument is about not
overclaiming.

This is a post-processor: it reads an existing run file, re-examines only the
rows that failed exact matching, and reports. It changes nothing upstream.

    python3 src/relocate.py runs/primary.jsonl
    python3 src/relocate.py runs/variance.jsonl --out runs/variance-relocated.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ami")
ROOT = os.path.abspath(os.path.expanduser(os.environ.get("AMI_ROOT", DEFAULT_ROOT)))
os.environ["AMI_ROOT"] = ROOT

import ami                                                     # noqa: E402

# Markup the corpus renders inline. A model reading the transcript sees these
# and reasonably leaves them out of a quote.
MARKUP = re.compile(r"^<[^>]*>$")

# Transcribed disfluency. Dropping these is normalisation, not invention.
FILLER = {"uh", "um", "erm", "er", "ah", "mm", "hmm", "eh", "mm-hmm", "uh-huh"}

WORD = re.compile(r"\S+")
EDGE_PUNC = re.compile(r"^[^\w<]+|[^\w>]+$")


def toks(text: str) -> list[tuple[str, int, int]]:
    """Normalised tokens with their spans in the ORIGINAL string.

    Three normalisations, each matching a thing models actually do:
      * drop inline markup      <disfmarker>, <other>, <laugh>
      * drop filler words       uh, um, mm
      * collapse a stutter      "the the" -> "the",  "I'm m" handled by edge-punc
    """
    out: list[tuple[str, int, int]] = []
    for m in WORD.finditer(text):
        raw = m.group(0)
        if MARKUP.match(raw):
            continue
        t = EDGE_PUNC.sub("", raw).lower()
        if not t or t in FILLER:
            continue
        if out and out[-1][0] == t:          # immediate repetition
            out[-1] = (t, out[-1][1], m.end())
            continue
        out.append((t, m.start(), m.end()))
    return out


def find(quote: str, doc: str, min_ratio: float = 0.70):
    """Locate `quote` in `doc` after normalisation. Returns (mode, span, ratio).

    mode is "normalised" for an exact token-sequence match, "fuzzy" for a
    close one, or None. The span returned always indexes the ORIGINAL doc, so
    a recovered citation still round-trips like any other.
    """
    q, d = toks(quote), toks(doc)
    if not q or not d:
        return None, None, 0.0

    qs = [t for t, _, _ in q]
    ds = [t for t, _, _ in d]

    # exact token-sequence match after normalisation
    n = len(qs)
    for i in range(len(ds) - n + 1):
        if ds[i:i + n] == qs:
            return "normalised", (d[i][1], d[i + n - 1][2]), 1.0

    # otherwise: how much of the quote appears, in order, anywhere in the doc
    sm = difflib.SequenceMatcher(None, ds, qs, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    if not blocks:
        return None, None, 0.0
    matched = sum(b.size for b in blocks)
    ratio = matched / len(qs)
    if ratio < min_ratio:
        return None, None, ratio
    lo = min(b.a for b in blocks)
    hi = max(b.a + b.size - 1 for b in blocks)
    return "fuzzy", (d[lo][1], d[hi][2]), ratio


# --------------------------------------------------------------------------- #
def corpus_for(trial: str, cache: dict) -> dict[str, str]:
    if trial not in cache:
        try:
            m = ami.load_meeting(f"{trial}a", ROOT)
            cache[trial] = {d.doc_id: d.text for d in m.docs.values()}
        except Exception:                                       # noqa: BLE001
            cache[trial] = {}
    return cache[trial]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="?", default="runs/primary.jsonl")
    ap.add_argument("--out", default=None,
                    help="write an augmented copy with match_mode/similarity")
    ap.add_argument("--min-ratio", type=float, default=0.70)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    path = Path(args.runs)
    if not path.exists():
        sys.exit(f"no run file at {path}")

    cache: dict[str, dict[str, str]] = {}
    stats: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"located": 0, "normalised": 0, "fuzzy": 0, "absent": 0})
    examples: list[tuple] = []
    out_rows: list[dict] = []

    for raw in path.open():
        raw = raw.strip()
        if not raw:
            continue
        try:
            r = json.loads(raw)
        except Exception:                                       # noqa: BLE001
            continue

        if r.get("kind") != "assertion":
            out_rows.append(r)
            continue

        key = (r.get("prompt_version", "?"), r.get("tier", "?"))

        if r.get("located"):
            stats[key]["located"] += 1
            r["match_mode"] = "exact"
            r["similarity"] = 1.0
            out_rows.append(r)
            continue

        corpus = corpus_for(str(r.get("trial", "")), cache)
        doc = corpus.get(str(r.get("doc_id", "")), "")
        mode, span, ratio = find(str(r.get("quote", "")), doc, args.min_ratio)

        if mode:
            stats[key][mode] += 1
            r["match_mode"] = mode
            r["similarity"] = round(ratio, 3)
            r["recovered_start"], r["recovered_end"] = span
            r["recovered_text"] = doc[span[0]:span[1]]
            if len(examples) < args.examples * 3:
                examples.append((mode, r.get("trial"), r.get("prompt_version"),
                                 r.get("quote", ""), r["recovered_text"]))
        else:
            stats[key]["absent"] += 1
            r["match_mode"] = "absent"
            r["similarity"] = round(ratio, 3)
        out_rows.append(r)

    # ---------------------------------------------------------------- report
    print(f"\n{path}")
    print(f"{'prompt':<18}{'tier':<9}{'exact':>7}{'normed':>8}{'fuzzy':>7}"
          f"{'ABSENT':>8}{'old FAB':>9}{'true FAB':>10}")
    print("-" * 76)

    tot = {"located": 0, "normalised": 0, "fuzzy": 0, "absent": 0}
    for key in sorted(stats):
        s = stats[key]
        for k in tot:
            tot[k] += s[k]
        n = sum(s.values())
        old = (n - s["located"]) / n if n else 0.0
        true = s["absent"] / n if n else 0.0
        print(f"{key[0]:<18}{key[1]:<9}{s['located']:>7}{s['normalised']:>8}"
              f"{s['fuzzy']:>7}{s['absent']:>8}{old:>9.3f}{true:>10.3f}")

    n = sum(tot.values())
    recovered = tot["normalised"] + tot["fuzzy"]
    print("-" * 76)
    if n:
        print(f"{'ALL':<18}{'':<9}{tot['located']:>7}{tot['normalised']:>8}"
              f"{tot['fuzzy']:>7}{tot['absent']:>8}"
              f"{(n - tot['located']) / n:>9.3f}{tot['absent'] / n:>10.3f}")
        unloc = n - tot["located"]
        if unloc:
            print(f"\n  Of {unloc} quotes that failed exact matching, "
                  f"{recovered} ({recovered / unloc:.1%}) are the right passage")
            print(f"  with the disfluencies cleaned up. {tot['absent']} are "
                  f"genuinely absent.")

    if examples:
        print("\n  RECOVERED -- what the model wrote vs what the transcript says")
        print("  " + "-" * 72)
        for mode, trial, prompt, q, got in examples[:args.examples]:
            print(f"  [{mode}] {trial} {prompt}")
            print(f"    model : {q[:150]}")
            print(f"    actual: {got[:150]}\n")

    print("  A recovered quote is NOT a pass. The model still did not copy "
          "what it\n  was asked to copy, and a citation nobody can verify "
          "byte for byte is\n  worth less than one they can. But it is not "
          "invention, and the two\n  must not share a number.\n")

    if args.out:
        o = Path(args.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        with o.open("w") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, default=str) + "\n")
        print(f"  wrote {o}  ({len(out_rows)} rows, with match_mode/similarity)\n")


if __name__ == "__main__":
    main()
