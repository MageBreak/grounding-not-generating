#!/usr/bin/env python3
"""Score a sweep against gold. The only program here that reads the answer key.

`runner.py` produces assertions and says whether each quote resolved. That much
needs no gold: an unlocated quote is fabricated whatever the right answer was.
This program adds the part that does need gold -- was the assertion correct --
and reports it as a curve over evidence availability rather than one number.

Gold is read from GOLD_PATH (default ~/gold/meetings.xml), outside the corpus
tree, so no AMI_ROOT path reaches it and nothing in the extraction path can.

    python3 src/score.py                      # runs/sweep.jsonl
    python3 src/score.py runs/sweep.jsonl --by tier
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prototype import load_gold   # noqa: E402  -- the scorer-side door


# --------------------------------------------------------------------------- #
class Bucket:
    __slots__ = ("asserted", "located", "correct", "refused", "misplaced",
                 "calls", "failed", "trials")

    def __init__(self):
        self.asserted = self.located = self.correct = 0
        self.refused = self.misplaced = self.calls = self.failed = 0
        self.trials: set[str] = set()

    @property
    def fabricated(self) -> int:
        """Asserted with a quote that is not in the document claimed."""
        return self.asserted - self.located

    @property
    def precision(self) -> float:
        """Of the claims it made, how many were right."""
        return self.correct / self.asserted if self.asserted else 0.0

    @property
    def grounded_precision(self) -> float:
        """Of the claims it could actually support, how many were right."""
        return self.correct / self.located if self.located else 0.0

    @property
    def fabrication_rate(self) -> float:
        return self.fabricated / self.asserted if self.asserted else 0.0

    @property
    def coverage(self) -> float:
        """Share of opportunities answered at all (assertions + refusals)."""
        total = self.asserted + self.refused
        return self.asserted / total if total else 0.0


def load_rows(path: Path):
    with path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    yield json.loads(raw)
                except Exception:                               # noqa: BLE001
                    continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs/sweep.jsonl")
    ap.add_argument("--by", nargs="+", default=["model", "prompt_version", "tier"],
                    help="grouping keys, in order")
    ap.add_argument("--sample", type=int, default=None,
                    help="score only this draw. Use --sample 0 on a "
                         "temperature-0 file: extra draws there are exact "
                         "duplicates and inflate n without adding information.")
    args = ap.parse_args()

    path = Path(args.runs)
    if not path.exists():
        sys.exit(f"no run store at {path} -- run src/runner.py first")

    gold = load_gold()
    buckets: dict[tuple, Bucket] = defaultdict(Bucket)
    unknown_trials: set[str] = set()

    for r in load_rows(path):
        if args.sample is not None and r.get("sample") != args.sample:
            continue
        key = tuple(str(r.get(k, "?")) for k in args.by)
        b = buckets[key]
        kind = r.get("kind")
        b.trials.add(str(r.get("trial", "?")))

        if kind == "call":
            b.calls += 1
            if not r.get("ok"):
                b.failed += 1
            continue

        if kind == "refusal":
            b.refused += 1
            continue

        if kind != "assertion":
            continue

        b.asserted += 1
        if r.get("misplaced_in"):
            b.misplaced += 1
        if not r.get("located"):
            continue
        b.located += 1

        # gold is keyed by the kickoff meeting id, e.g. ES2002a
        mid = f"{r.get('trial', '')}a"
        truth = gold.get(mid)
        if truth is None:
            unknown_trials.add(mid)
            continue
        if truth.get(r.get("speaker", "")) == r.get("role"):
            b.correct += 1

    hdr = "  ".join(f"{k:<17}" for k in args.by)
    print(f"\n{hdr} {'trls':>4} {'calls':>5} {'fail':>4} {'asrt':>5} "
          f"{'loc':>5} {'FAB':>6} {'refus':>5} {'prec':>6} {'gprec':>6} "
          f"{'cov':>6}")
    print("-" * (len(hdr) + 62))

    for key in sorted(buckets):
        b = buckets[key]
        label = "  ".join(f"{k:<17}" for k in key)
        print(f"{label} {len(b.trials):>4} {b.calls:>5} {b.failed:>4} "
              f"{b.asserted:>5} {b.located:>5} {b.fabrication_rate:>6.3f} "
              f"{b.refused:>5} {b.precision:>6.3f} "
              f"{b.grounded_precision:>6.3f} {b.coverage:>6.3f}")

    # Unequal effort across conditions is a comparability problem, not a
    # detail. Say so here rather than letting a reader assume balance.
    percall = {k: b.calls / max(len(b.trials), 1) for k, b in buckets.items()}
    if percall and (max(percall.values()) - min(percall.values())) > 0.5:
        print("\n  ** UNBALANCED: conditions have different numbers of calls "
              "per trial.\n     Re-run with --sample 0, or finish the sweep, "
              "before comparing rows.")

    print("""
  FAB    fabrication rate -- share of assertions whose quote is not in the
         document it was attributed to. Needs no gold; it is a property of
         the output alone.
  prec   correct / asserted
  gprec  correct / located -- precision among claims the model could support.
         gprec well above prec is the finding: the quote test is separating
         what the model read from what it supplied.
  cov    asserted / (asserted + refused)
""")

    misplaced = sum(b.misplaced for b in buckets.values())
    if misplaced:
        print(f"  {misplaced} quote(s) existed, but in a different document "
              f"than claimed.\n  That is the speaker-letter trap one level up "
              f"-- count it separately.\n")
    if unknown_trials:
        print(f"  no gold for: {', '.join(sorted(unknown_trials))}\n")

    print("  baseline for comparison (prototype.py, regex, same task):")
    print("    grounded  precision 0.970  coverage 0.493")
    print("    naive     precision 0.610  coverage 1.000   fabrication 0.507\n")


if __name__ == "__main__":
    main()
