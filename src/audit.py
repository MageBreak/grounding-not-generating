#!/usr/bin/env python3
"""Stage 6: the isolated grounding audit.

Why this exists, in one line from our own data: in the `none` evidence tier the
quote test passed on 100% of assertions while precision sat at 0.395. The model
does not invent text -- it reaches for real nearby text and attaches it to a
guess, and a string search cannot tell the difference.

So a second model is asked one narrow question:

    Does THIS passage state THIS claim?

It is given the span and the claim and nothing else. Not the transcript, not
the speaker labels, not the other candidate spans, not the extractor's
reasoning. `num_ctx` is set small enough that it could not hold the document
even if something tried to pass it -- the isolation is structural, not a
promise in a prompt.

Two properties this must keep, or it is worth nothing:

  * It can only SUBTRACT. There is no verdict that adds a claim, marks one
    correct, or satisfies a gate. The most it can do is remove.
  * It must not be the same model that produced the claim, or it is grading
    its own homework. --model defaults to a different family from the sweep.

It never reads gold. `score.py --audit` evaluates how well it did.

    python3 src/audit.py runs/primary.jsonl --model ollama:gemma3:12b
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import Call, backend_for                             # noqa: E402

ROLE_WORDS = {
    "PM": "the project manager",
    "ME": "the marketing expert",
    "UI": "the user interface designer",
    "ID": "the industrial designer",
}

SYSTEM = (
    "You judge whether a single passage states a single claim.\n\n"
    "You will be shown one PASSAGE and one CLAIM. Decide whether the passage "
    "itself states the claim.\n\n"
    "Answer `supported` only if the passage says it. Answer `unsupported` if "
    "the passage is about something else, is too vague, merely mentions a "
    "related word, or would require outside knowledge or an assumption to "
    "connect it to the claim.\n\n"
    "The speaker being the one who said the passage is not evidence about who "
    "they are. A passage that would be true of anyone in the room supports "
    "nothing in particular.\n\n"
    "You do not have the rest of the document and you are not meant to. Judge "
    "only what is in front of you. `unsupported` is the correct answer "
    "whenever the passage alone does not settle it."
)

VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "unsupported"]},
        "why": {"type": "string", "description": "one short sentence"},
    },
    "required": ["verdict", "why"],
}


def claim_text(row: dict) -> str:
    role = ROLE_WORDS.get(row.get("role", ""), row.get("role", "?"))
    return f"The person speaking in this passage is {role}."


def audit_key(row: dict) -> str:
    return f"{row.get('run', '')}|{row.get('speaker', '')}"


def load_assertions(paths: list[Path]):
    """Every located assertion across the given run files.

    Unlocated ones are not sent: the quote does not exist, the deterministic
    layer already rejected them, and there is nothing for an auditor to read.
    """
    for p in paths:
        with p.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    r = json.loads(raw)
                except Exception:                               # noqa: BLE001
                    continue
                if r.get("kind") == "assertion" and r.get("located"):
                    yield r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", default=["runs/primary.jsonl"])
    ap.add_argument("--model", default="ollama:gemma3:12b",
                    help="MUST differ from the extractor's model")
    ap.add_argument("--out", default="runs/audit.jsonl")
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="deliberately small: too small to hold a transcript")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = [Path(p) for p in args.runs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"no such run file: {missing[0]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open() as fh:
            for raw in fh:
                try:
                    done.add(json.loads(raw)["key"])
                except Exception:                               # noqa: BLE001
                    pass

    rows = list(load_assertions(paths))
    todo = [r for r in rows if audit_key(r) not in done]
    if args.limit:
        todo = todo[:args.limit]

    same_model = {r.get("model") for r in rows} & {args.model}
    print(f"auditor   {args.model}")
    print(f"assertions {len(rows)} located, {len(todo)} to audit")
    print(f"num_ctx   {args.num_ctx} (cannot hold a transcript)\n")
    if same_model:
        print("  ** WARNING: the auditor is the same model that produced these")
        print("     claims. It is grading its own work. Use another family.\n")

    backend = backend_for(args.model)
    counts = {"supported": 0, "unsupported": 0, "failed": 0}

    with out.open("a") as fh:
        for i, r in enumerate(todo, 1):
            span = r.get("exact_text", "")
            claim = claim_text(r)
            call = Call(
                model=args.model,
                system=SYSTEM,
                # The passage and the claim. Nothing identifies the document,
                # the speaker, or where in the meeting this came from.
                user=f"PASSAGE:\n{span}\n\nCLAIM:\n{claim}",
                prompt_version="audit-v1",
                schema=VERDICT_SCHEMA,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                max_tokens=160,
                think=False,
            )
            resp = backend.complete(call)

            verdict, why = "error", (resp.error or "")[:200]
            if resp.ok:
                try:
                    d = resp.parsed()
                    verdict = str(d.get("verdict", "error"))
                    why = str(d.get("why", ""))[:200]
                except Exception as e:                          # noqa: BLE001
                    why = f"unparseable: {e}"

            # The monotonicity rule, enforced here rather than assumed: the
            # only effect an audit row may have downstream is removal.
            removed = verdict == "unsupported"
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict == "error":
                counts["failed"] += 1

            fh.write(json.dumps({
                "kind": "audit",
                "key": audit_key(r),
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "auditor": args.model,
                "prompt_version": "audit-v1",
                "run": r.get("run"), "trial": r.get("trial"),
                "tier": r.get("tier"),
                "extractor": r.get("model"),
                "extractor_prompt": r.get("prompt_version"),
                "speaker": r.get("speaker"), "role": r.get("role"),
                "span_len": len(span), "ambiguous": r.get("ambiguous"),
                "verdict": verdict, "removed": removed, "why": why,
                "latency_s": round(resp.latency_s, 2),
            }) + "\n")
            fh.flush()

            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] supported={counts['supported']} "
                      f"unsupported={counts['unsupported']} "
                      f"failed={counts['failed']}")

    n = counts["supported"] + counts["unsupported"]
    print(f"\n  audited {n}   removed {counts['unsupported']}"
          + (f" ({counts['unsupported'] / n:.1%})" if n else ""))
    print(f"\n  The audit can only subtract. Whether it subtracted the RIGHT")
    print(f"  ones needs gold:  python3 src/score.py <runs> --audit {out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  interrupted. Completed verdicts are saved -- re-run to resume.")
        sys.exit(130)
