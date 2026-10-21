"""Provider shim + the locate() mechanism.

Two jobs, deliberately in one file because they are two halves of one contract:

  1. `Backend` -- one interface over a local Ollama model and a hosted API, so a
     sweep can vary the model without touching a call site.
  2. `locate()` -- the model returns a *quote*; this module finds it in the
     corpus. A quote that cannot be located was not read. It was invented.

Nothing here grades model output. That is schema.py's job, and it is
deterministic.

Standard library only, except for the two optional backends:
    pip install ollama            # local
    pip install google-genai      # hosted

Self-test:  python3 llm.py
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from schema import Citation

# --------------------------------------------------------------------------- #
# The call and its result -- both are records, both are serialisable.
#
# These line up with the ArkenOS ontology's `Run` and `Experience`: a Call is
# the configuration of one transactionally distinct attempt, a Response is the
# immutable record of what came back. Two identical Calls are still two
# occurrences, which is why `sample` is part of the key.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Call:
    model: str                      # "ollama:qwen3:14b" | "gemini:gemini-2.5-flash"
    system: str
    user: str
    prompt_version: str             # bump this whenever the wording changes
    schema: dict[str, Any] | None = None   # JSON Schema -> structured output
    temperature: float = 0.0
    seed: int = 0
    sample: int = 0                 # nth draw of an otherwise identical cell
    num_ctx: int = 32768            # MUST exceed the input. See guard below.
    max_tokens: int = 2048
    # Reasoning mode on a thinking-capable model. None = provider default,
    # which is NOT good enough for an experiment: whether the model reasons
    # before answering is a condition, and an uncontrolled condition is a
    # confound. The runner always sets this explicitly.
    think: bool | None = None

    def key(self) -> str:
        """Cache key. Everything that can change the answer is in here."""
        import hashlib
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Response:
    call_key: str
    model: str
    text: str
    ok: bool
    latency_s: float
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # "stop" = the model finished. "length" = it was cut off at max_tokens,
    # which produces truncated JSON. A truncated generation is a harness
    # failure, not a fabrication, and the two must never be pooled.
    done_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def parsed(self) -> Any:
        """Structured output, parsed. Raises if the model ignored the schema."""
        return json.loads(self.text)


class ContextOverflow(RuntimeError):
    """Raised instead of letting the runtime silently truncate the prompt.

    This is the most dangerous failure mode in the whole harness: a truncated
    transcript makes a *correct* refusal indistinguishable from a model that
    never saw the evidence. It would show up in results as a grounding failure
    and it would be our bug, not the model's.
    """


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str) -> int:
    """Deliberately crude and deliberately pessimistic.

    ~3.2 chars/token is conservative for English transcripts; we would rather
    refuse a prompt that would have fit than truncate one that did not.
    """
    return int(len(text) / 3.2) + 1


class Backend(Protocol):
    name: str

    def complete(self, call: Call) -> Response: ...


class OllamaBackend:
    """Local inference. Model string: "ollama:<tag>", e.g. "ollama:qwen3:14b"."""

    def __init__(self, tag: str, host: str | None = None):
        self.tag = tag
        self.name = f"ollama:{tag}"
        self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    def complete(self, call: Call) -> Response:
        import urllib.request

        need = estimate_tokens(call.system + call.user) + call.max_tokens
        if need > call.num_ctx:
            raise ContextOverflow(
                f"prompt needs ~{need} tokens, num_ctx is {call.num_ctx}. "
                f"Raise num_ctx or shorten the input -- do not let it truncate."
            )

        body: dict[str, Any] = {
            "model": self.tag,
            "messages": [
                {"role": "system", "content": call.system},
                {"role": "user", "content": call.user},
            ],
            "stream": False,
            "options": {
                "temperature": call.temperature,
                "seed": call.seed + call.sample,
                # num_ctx is the whole ballgame. Ollama's default is small
                # (2048-4096 depending on version) and it truncates in silence.
                "num_ctx": call.num_ctx,
                "num_predict": call.max_tokens,
            },
        }
        if call.schema:
            body["format"] = call.schema
        if call.think is not None:
            body["think"] = call.think

        t0 = time.time()
        raw = None
        for attempt in (1, 2):
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    raw = json.loads(r.read())
                break
            except Exception as e:                              # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                # Models without reasoning support reject the field outright.
                # Drop it once and retry, so a mixed-model sweep does not die
                # on its first non-thinking model.
                if attempt == 1 and "think" in body and "think" in msg.lower():
                    body.pop("think")
                    continue
                return Response(call.key(), self.name, "", False,
                                time.time() - t0, error=msg)

        return Response(
            call_key=call.key(),
            model=self.name,
            text=raw.get("message", {}).get("content", ""),
            ok=True,
            latency_s=time.time() - t0,
            prompt_tokens=raw.get("prompt_eval_count"),
            completion_tokens=raw.get("eval_count"),
            done_reason=raw.get("done_reason"),
            raw={k: v for k, v in raw.items() if k != "message"},
        )


class GeminiBackend:
    """Hosted. Model string: "gemini:<model>". Key from GEMINI_API_KEY.

    The key is read from the environment and never logged. Do not pass it on a
    command line -- argv is visible to every process on the machine and lands
    in agent transcripts.
    """

    def __init__(self, model: str):
        self.model = model
        self.name = f"gemini:{model}"

    def complete(self, call: Call) -> Response:
        from google import genai
        from google.genai import types

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return Response(call.key(), self.name, "", False, 0.0,
                            error="GEMINI_API_KEY not set")

        cfg: dict[str, Any] = {
            "system_instruction": call.system,
            "temperature": call.temperature,
            "max_output_tokens": call.max_tokens,
        }
        if call.schema:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = call.schema

        t0 = time.time()
        try:
            client = genai.Client(api_key=key)
            r = client.models.generate_content(
                model=self.model,
                contents=call.user,
                config=types.GenerateContentConfig(**cfg),
            )
        except Exception as e:                                  # noqa: BLE001
            return Response(call.key(), self.name, "", False,
                            time.time() - t0, error=f"{type(e).__name__}: {e}")

        usage = getattr(r, "usage_metadata", None)
        return Response(
            call_key=call.key(),
            model=self.name,
            text=r.text or "",
            ok=True,
            latency_s=time.time() - t0,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
        )


def backend_for(model: str) -> Backend:
    """"ollama:qwen3:14b" -> OllamaBackend("qwen3:14b")."""
    provider, _, rest = model.partition(":")
    if provider == "ollama":
        return OllamaBackend(rest)
    if provider == "gemini":
        return GeminiBackend(rest)
    raise ValueError(f"unknown provider in {model!r} (expected ollama: or gemini:)")


# --------------------------------------------------------------------------- #
# locate() -- the mechanism the whole claim rests on
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse runs of whitespace, keeping a map back to original offsets.

    A model that read the transcript will reproduce the words but not
    necessarily the spacing -- transcripts are full of newlines and double
    spaces. Matching on normalised text with an offset map lets us be generous
    about whitespace and unforgiving about everything else.

    Returns (normalised_text, index_map) where index_map[i] is the offset in
    the original string of normalised_text[i].
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True            # leading whitespace is dropped
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    # trailing single space, if any
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


@dataclass
class LocateResult:
    citation: Citation | None
    count: int                   # how many times the quote occurs
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.citation is not None

    @property
    def ambiguous(self) -> bool:
        return self.count > 1


def locate(quote: str, doc_id: str, corpus: dict[str, str],
           case_sensitive: bool = True) -> LocateResult:
    """Find `quote` in corpus[doc_id] and return a verified Citation.

    The returned Citation always satisfies the project contract:

        corpus[doc_id][start:end] == citation.exact_text

    exact_text is the *original* substring, not the model's rendering of it, so
    a quote that differs only in whitespace still produces a citation that
    round-trips byte for byte.

    A LocateResult with found=False is the signal that matters: the model
    produced text it attributed to the document, and the document does not
    contain it.
    """
    doc = corpus.get(doc_id)
    if doc is None:
        return LocateResult(None, 0, f"no such document: {doc_id}")

    q = quote.strip()
    if not q:
        return LocateResult(None, 0, "empty quote")

    norm_doc, idx = _normalise(doc)
    norm_q, _ = _normalise(q)

    hay = norm_doc if case_sensitive else norm_doc.lower()
    needle = norm_q if case_sensitive else norm_q.lower()

    count, pos, first = 0, 0, -1
    while (pos := hay.find(needle, pos)) != -1:
        if first < 0:
            first = pos
        count += 1
        pos += 1
    if first < 0:
        return LocateResult(None, 0, "quote not present in document")

    start = idx[first]
    end = idx[first + len(norm_q) - 1] + 1
    cite = Citation(doc_id, start, end, doc[start:end])

    # Belt and braces: never hand back a citation that does not round-trip.
    assert cite.verify(corpus), (doc_id, start, end)
    return LocateResult(cite, count)


def locate_anywhere(quote: str, corpus: dict[str, str]) -> LocateResult:
    """Same, but search every document. Use when the model names no doc_id.

    An unambiguous hit in exactly one document is worth more than a hit the
    model told us where to find -- it did not get to choose.
    """
    hits: list[LocateResult] = []
    for doc_id in corpus:
        r = locate(quote, doc_id, corpus)
        if r.found:
            hits.append(r)
    if not hits:
        return LocateResult(None, 0, "quote not present in any document")
    total = sum(h.count for h in hits)
    return LocateResult(hits[0].citation, total,
                        "" if total == 1 else f"occurs {total}x across {len(hits)} docs")


# --------------------------------------------------------------------------- #
# A schema the model is forced to fill. Quotes in, spans out.
# --------------------------------------------------------------------------- #

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "quote": {
                        "type": "string",
                        "description": "Verbatim span from the document that "
                                       "states this. Copy exactly. Do not "
                                       "paraphrase, summarise or join passages.",
                    },
                    "doc_id": {"type": "string"},
                },
                "required": ["subject", "predicate", "object", "quote", "doc_id"],
            },
        },
        "refusals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["question", "why"],
            },
            "description": "Things asked for that the document does not state. "
                           "Putting an item here is always better than guessing.",
        },
    },
    "required": ["claims", "refusals"],
}


def ground(response: Response, corpus: dict[str, str]) -> dict[str, Any]:
    """Resolve every quote in a structured response. No judgement, just facts.

    Returns the claims split three ways:
      grounded   -- quote located, citation verified
      unlocated  -- quote not in the document it was attributed to
      misplaced  -- quote not in that document, but present in another
    """
    out: dict[str, list[Any]] = {"grounded": [], "unlocated": [], "misplaced": []}
    try:
        payload = response.parsed()
    except Exception as e:                                      # noqa: BLE001
        return {**out, "parse_error": f"{type(e).__name__}: {e}"}

    for c in payload.get("claims", []):
        r = locate(c.get("quote", ""), c.get("doc_id", ""), corpus)
        if r.found:
            out["grounded"].append({**c, "citation": r.citation,
                                    "ambiguous": r.ambiguous})
            continue
        elsewhere = locate_anywhere(c.get("quote", ""), corpus)
        if elsewhere.found:
            out["misplaced"].append({**c, "actually_in": elsewhere.citation.doc_id,
                                     "citation": elsewhere.citation})
        else:
            out["unlocated"].append({**c, "reason": r.reason})
    out["refusals"] = payload.get("refusals", [])
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    corpus = {
        "ES2002a.C": "Um I'm Craig and I'm\n  User   Interface.",
        "ES2002a.D": "And I'm Andrew and I'm uh our marketing expert.",
    }

    print("locate() -- exact")
    r = locate("Craig", "ES2002a.C", corpus)
    print("   ", r.citation, "->", repr(corpus["ES2002a.C"][r.citation.start:r.citation.end]))

    print("locate() -- whitespace differs from the model's rendering")
    r = locate("I'm User Interface.", "ES2002a.C", corpus)
    print("   ", r.citation, "->", repr(r.citation.exact_text))
    assert r.citation.verify(corpus), "round-trip must hold on the ORIGINAL text"

    print("locate() -- fabricated quote")
    r = locate("I'm the project manager.", "ES2002a.C", corpus)
    print("   ", r.found, r.reason)

    print("locate() -- right words, wrong document")
    r = locate("Andrew", "ES2002a.C", corpus)
    e = locate_anywhere("Andrew", corpus)
    print("    attributed:", r.found, "| actually in:", e.citation.doc_id)

    print("\ncontext guard")
    try:
        OllamaBackend("qwen3:14b").complete(
            Call(model="ollama:qwen3:14b", system="", user="x" * 400_000,
                 prompt_version="v0", num_ctx=8192))
    except ContextOverflow as e:
        print("    refused:", str(e)[:70], "...")
