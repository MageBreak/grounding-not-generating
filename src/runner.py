#!/usr/bin/env python3
"""The sweep runner: stages 1-7 of the pipeline, resumable, over one task.

The task is deliberately the same one `prototype.py` measures -- who holds
which role, from the kickoff meeting of each AMI scenario trial -- so that a
model's numbers sit directly beside the regex baseline rather than beside
nothing.

What this file does NOT do, on purpose:

  * It never reads gold. Not once, not for tier labelling, not for early
    stopping. `score.py` reads gold and this does not import it.
  * It never judges. It records what the model asserted and whether each quote
    resolved to a span. Whether the assertion was right is a later, separate
    question asked by a different program.

Output is one JSONL file of `Experience` rows -- immutable records of what was
observed in a `Run` -- appended as they happen, so a crash costs you nothing
and a re-run skips what is already there.

    python3 src/runner.py --models ollama:qwen3:14b --samples 3
    python3 src/runner.py --models ollama:qwen3:14b --trials ES2002 --prompts p2_cite_or_refuse
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ami")
ROOT = os.path.abspath(os.path.expanduser(os.environ.get("AMI_ROOT", DEFAULT_ROOT)))
os.environ["AMI_ROOT"] = ROOT

import ami                                                    # noqa: E402
from llm import Call, backend_for, locate, locate_anywhere    # noqa: E402

# `prototype` reads sys.argv at import time; borrow argv so it resolves ROOT
# the same way we did. We import it for one reason only: the definition of
# "a speaker states their own role" must be byte-identical to the baseline's,
# or the evidence tiers are not comparable.
_argv = sys.argv
sys.argv = [sys.argv[0], ROOT]
import prototype as _proto                                    # noqa: E402
sys.argv = _argv

ROLES = {
    "PM": "project manager",
    "ME": "marketing expert",
    "UI": "user interface designer",
    "ID": "industrial designer",
}

# --------------------------------------------------------------------------- #
# The schema the model must fill. Constant across every prompt condition --
# only the instructions vary, so the instruction is the only variable.
# --------------------------------------------------------------------------- #

NOT_STATED = "not_stated"


def task_schema(speakers: list[str]) -> dict:
    """One slot per speaker, keyed by the speaker's own label.

    An earlier version used a free `assertions` array. A model given an array
    fills it: one run produced nineteen entries for four speakers, several of
    them quoting "Okay.", and put the ROLE in the speaker field. Both were
    structural invitations, so both are fixed structurally rather than by
    asking more nicely -- the shape now admits exactly one answer per speaker
    and the speaker labels are the object's own keys, which the model cannot
    invent.
    """
    per_speaker = {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": list(ROLES) + [NOT_STATED],
                "description": f"{NOT_STATED} if the transcript does not say",
            },
            "doc_id": {
                "type": "string",
                "description": "the [bracketed] id of the line the quote is "
                               f"from; empty string if {NOT_STATED}",
            },
            "quote": {
                "type": "string",
                "description": "verbatim span from that line stating the "
                               f"role. Copy exactly. Empty if {NOT_STATED}.",
            },
        },
        "required": ["role", "doc_id", "quote"],
    }
    return {
        "type": "object",
        "properties": {s: dict(per_speaker) for s in speakers},
        "required": list(speakers),
    }

ROLE_LIST = ", ".join(f"{k} ({v})" for k, v in ROLES.items())

_PREAMBLE = (
    "You are reading a transcript of a project kick-off meeting. Each line is "
    "prefixed, in square brackets, with the id of the document it came from.\n\n"
    f"The four roles are: {ROLE_LIST}.\n\n"
    "Answer with one entry per speaker, keyed by the speaker's letter.\n"
)

PROMPTS: dict[str, str] = {
    # Maximum completion pressure -- the LLM analogue of extract_naive().
    # `not_stated` stays available in the schema so that the instruction, not
    # the shape of the output, is the only thing that changes between
    # conditions. The quote is still required, so a guess has to come with a
    # fabricated quote, and locate() catches that.
    "p0_complete": _PREAMBLE + (
        "\nGive every speaker a role. Produce a complete answer: do not leave "
        "anyone unassigned. Fill in the quote and doc_id fields as well."
    ),
    # Citation required, no permission to decline.
    "p1_cite": _PREAMBLE + (
        "\nFor each speaker, give their role and the verbatim quote from the "
        "transcript that states it, with the doc_id of the line it came from. "
        "Copy the quote exactly as it appears -- do not paraphrase, tidy, or "
        "join separate passages."
    ),
    # Citation required and refusal made legitimate.
    "p2_cite_or_refuse": _PREAMBLE + (
        "\nGive a speaker a role ONLY where the transcript states it, together "
        "with the verbatim quote and its doc_id, copied exactly -- do not "
        f"paraphrase, tidy, or join separate passages.\n\n"
        f"Where the transcript does not state a speaker's role, set their role "
        f"to \"{NOT_STATED}\" and leave quote and doc_id empty. That is a "
        f"correct answer, not a failure. Guessing from convention, from the "
        f"order people speak, or from what you know about meetings like this "
        f"one, is not.\n\n"
        "The quote must be the passage in which that speaker's role is stated. "
        "A line that merely happens to be spoken by them is not evidence of "
        "their role."
    ),
}


# --------------------------------------------------------------------------- #
# Stage 1 -- the passages
# --------------------------------------------------------------------------- #


def transcript(meeting) -> str:
    """Interleaved turns, each tagged with the document it lives in.

    Interleaving matters: a role claim can be split across an interruption, and
    a view that groups each speaker's words together hides that. Tagging each
    line with its doc_id keeps both properties -- the reader sees the real
    sequence, and a quote still resolves inside one document, because the
    interrupting speaker's words live in a different file entirely.
    """
    lines = []
    for u in meeting.utterances:
        t = u.text.strip()
        if t:
            lines.append(f"[{u.doc_id}] {t}")
    return "\n".join(lines)


def corpus_of(meeting) -> dict[str, str]:
    return {d.doc_id: d.text for d in meeting.docs.values()}


def evidence_tier(meeting) -> tuple[str, int]:
    """How much role evidence this meeting actually contains.

    Uses the baseline's own extractor, so "stated" means exactly what it meant
    in prototype.py. Computed from the text, never from gold -- a tier label
    derived from the answer key would leak it straight back in.
    """
    stated, _questions, _cites = _proto.extract_grounded(meeting)
    n = len({k for k, v in stated.items() if v})
    tier = "full" if n >= 4 else ("none" if n == 0 else "partial")
    return tier, n


# --------------------------------------------------------------------------- #
# Run records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """One transactionally distinct attempt. Two identical cells differing only
    in `sample` are still two occurrences -- that is why sample is in the key."""
    model: str
    prompt_version: str
    trial: str
    sample: int
    think: bool
    temperature: float
    num_ctx: int
    max_tokens: int

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


class RunStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if self.path.exists():
            with self.path.open() as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                    except Exception:                           # noqa: BLE001
                        continue
                    if d.get("kind") == "call" and d.get("ok"):
                        self.done.add(d.get("run", ""))

    def seen(self, cell: Cell) -> bool:
        return cell.key() in self.done

    def append(self, rows: list[dict]) -> None:
        with self.path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        for r in rows:
            if r.get("kind") == "call" and r.get("ok"):
                self.done.add(r.get("run", ""))


# --------------------------------------------------------------------------- #
# Stages 2-3 -- generate, then resolve
# --------------------------------------------------------------------------- #


def run_cell(cell: Cell, meeting, tier: str, n_stated: int) -> list[dict]:
    corpus = corpus_of(meeting)
    body = transcript(meeting)
    speakers = sorted(meeting.docs)

    call = Call(
        model=cell.model,
        system=PROMPTS[cell.prompt_version],
        user=body,
        prompt_version=cell.prompt_version,
        schema=task_schema(speakers),
        temperature=cell.temperature,
        seed=0,
        sample=cell.sample,
        num_ctx=cell.num_ctx,
        think=cell.think,
        max_tokens=cell.max_tokens,
    )

    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    common = {
        "run": cell.key(), "ts": stamp, "model": cell.model,
        "prompt_version": cell.prompt_version, "trial": cell.trial,
        "tier": tier, "n_stated": n_stated, "sample": cell.sample,
        "think": cell.think, "num_ctx": cell.num_ctx,
        "temperature": cell.temperature, "max_tokens": cell.max_tokens,
    }

    try:
        resp = backend_for(cell.model).complete(call)
    except Exception as e:                                      # noqa: BLE001
        return [{**common, "kind": "call", "ok": False,
                 "error": f"{type(e).__name__}: {e}"}]

    rows: list[dict] = [{
        **common, "kind": "call", "ok": resp.ok, "error": resp.error,
        "latency_s": round(resp.latency_s, 2),
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "done_reason": resp.done_reason,
        "speakers": speakers,
    }]
    if not resp.ok:
        return rows

    try:
        payload = resp.parsed()
    except Exception as e:                                      # noqa: BLE001
        rows[0]["ok"] = False
        # Truncation and malformed output are different failures. Pooling them
        # would let a max_tokens setting masquerade as model behaviour.
        kind = "truncated" if resp.done_reason == "length" else "unparseable"
        rows[0]["error"] = f"{kind}: {type(e).__name__}: {e}"
        rows[0]["raw_text"] = resp.text[-600:]
        return rows

    for spk in speakers:
        a = payload.get(spk) or {}
        role = str(a.get("role", "")).strip()
        quote = str(a.get("quote", "")).strip()
        doc_id = str(a.get("doc_id", "")).strip().strip("[]")

        if role == NOT_STATED or (not role):
            rows.append({**common, "kind": "refusal", "speaker": spk,
                         "why": quote[:200] or "role not stated"})
            continue

        r = locate(quote, doc_id, corpus) if quote else None
        row = {
            **common, "kind": "assertion", "speaker": spk, "role": role,
            "doc_id": doc_id, "quote": quote,
            "located": bool(r and r.found),
            "ambiguous": bool(r and r.ambiguous),
        }
        if r and r.found:
            row.update(start=r.citation.start, end=r.citation.end,
                       exact_text=r.citation.exact_text)
        else:
            # Right words, wrong document is its own failure -- the same
            # mistake as the speaker-letter trap, one level up. Keep it
            # distinguishable from a quote that exists nowhere at all.
            e = locate_anywhere(quote, corpus) if quote else None
            row["misplaced_in"] = e.citation.doc_id if (e and e.found) else None
            row["reason"] = (r.reason if r else "no quote given")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
def trials_available() -> list[str]:
    wordsdir = os.path.join(ROOT, "words")
    return sorted({f[:6] for f in os.listdir(wordsdir)
                   if f[:2] in ("ES", "IS", "TS")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["ollama:qwen3:14b"])
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS),
                    choices=list(PROMPTS))
    ap.add_argument("--trials", nargs="+", default=None,
                    help="default: every scenario trial")
    ap.add_argument("--samples", type=int, default=1,
                    help="draws per cell; >=5 if you want error bars")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="generation cap; truncation is recorded, not guessed at")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true",
                    help="reasoning mode on (recorded as a condition)")
    ap.add_argument("--out", default="runs/sweep.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and print the plan, call nothing")
    args = ap.parse_args()

    trials = args.trials or trials_available()
    store = RunStore(Path(args.out))

    cells = [Cell(m, p, t, s, args.think, args.temperature,
                  args.num_ctx, args.max_tokens)
             for m in args.models for p in args.prompts
             for t in trials for s in range(args.samples)]
    todo = [c for c in cells if not store.seen(c)]

    print(f"AMI_ROOT  {ROOT}")
    print(f"out       {args.out}")
    print(f"cells     {len(cells)}  ({len(cells) - len(todo)} already done, "
          f"{len(todo)} to run)")
    print(f"models    {', '.join(args.models)}")
    print(f"prompts   {', '.join(args.prompts)}")
    print(f"think     {args.think}   num_ctx {args.num_ctx}   "
          f"temp {args.temperature}\n")

    cache: dict[str, tuple] = {}
    counts = {"assertion": 0, "located": 0, "refusal": 0, "failed": 0}

    for i, cell in enumerate(todo, 1):
        mid = f"{cell.trial}a"
        if mid not in cache:
            try:
                m = ami.load_meeting(mid, ROOT)
            except Exception as e:                              # noqa: BLE001
                print(f"  [{i}/{len(todo)}] {mid}: load failed -- {e}")
                continue
            if not m.docs:
                print(f"  [{i}/{len(todo)}] {mid}: no documents, skipped")
                continue
            cache[mid] = (m, *evidence_tier(m))
        meeting, tier, n_stated = cache[mid]

        if args.dry_run:
            body = transcript(meeting)
            from llm import estimate_tokens
            print(f"  [{i}/{len(todo)}] {cell.trial} {cell.prompt_version} "
                  f"s{cell.sample}  tier={tier}({n_stated})  "
                  f"~{estimate_tokens(body)} tok")
            continue

        rows = run_cell(cell, meeting, tier, n_stated)
        store.append(rows)

        a = [r for r in rows if r["kind"] == "assertion"]
        loc = [r for r in a if r.get("located")]
        ref = [r for r in rows if r["kind"] == "refusal"]
        call = rows[0]
        counts["assertion"] += len(a)
        counts["located"] += len(loc)
        counts["refusal"] += len(ref)
        if not call.get("ok"):
            counts["failed"] += 1

        status = "ok" if call.get("ok") else f"FAIL {call.get('error', '')[:40]}"
        print(f"  [{i}/{len(todo)}] {cell.trial} {cell.prompt_version} "
              f"s{cell.sample} tier={tier:<7} "
              f"asserted={len(a)} located={len(loc)} refused={len(ref)} "
              f"{call.get('latency_s', 0)}s {status}")

    if not args.dry_run:
        n = counts["assertion"]
        unloc = n - counts["located"]
        print(f"\n  assertions {n}   located {counts['located']}   "
              f"UNLOCATED {unloc}" + (f" ({unloc / n:.1%})" if n else ""))
        print(f"  refusals   {counts['refusal']}   failed calls {counts['failed']}")
        print(f"\n  An unlocated quote is a fabricated one. That number needs no")
        print(f"  gold labels. For precision and coverage:  python3 src/score.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Everything completed is already on disk; re-running skips it.
        print("\n  interrupted. Finished cells are saved -- re-run to resume.")
        sys.exit(130)
