# Running the model locally

Target machine: MSI Vector 16 HX, **RTX 5080 Laptop (16 GB VRAM)**, 24 CPU
threads, ~16 GB system RAM, Linux.

The sweep runs as **your own process, calling Ollama directly**. ArkenOS agents
do engineering work; they are not in the measurement path. This is not a style
preference — everything inside ArkenOS runs as a child of an Electron app with
no crash-durable execution, so a six-hour run belongs outside that process tree
(`setsid nohup`), writing JSONL you can resume from.

---

## 1. Install

```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl --user status ollama      # or: ollama serve
nvidia-smi                          # confirm the GPU is visible
```

Environment, set before `ollama serve` (put these in `~/.config/environment.d/`
or your unit file, not in the repo):

```bash
OLLAMA_FLASH_ATTENTION=1     # lower KV-cache cost; worth having at long context
OLLAMA_KV_CACHE_TYPE=q8_0    # halves KV memory vs fp16, negligible quality cost
OLLAMA_KEEP_ALIVE=30m        # stop it unloading the model between sweep calls
OLLAMA_NUM_PARALLEL=1        # see the warning in §3
```

---

## 2. Model choice at 16 GB

Budget: **weights + KV cache must both fit**, or layers spill to CPU and
throughput collapses. Rough guide, Q4_K_M quantisation:

| Model | Weights | Why it's on the list |
|---|---:|---|
| `qwen3:14b` | ~9 GB | Default. Strong instruction-following, reliable JSON-schema adherence. |
| `qwen3:8b` | ~5 GB | Same family, much more context headroom. Use when a trial is long. |
| `gemma3:12b` | ~8 GB | Different family — needed for a cross-family claim. |
| `llama3.1:8b` | ~5 GB | The weak baseline. Useful precisely because it's weaker. |
| `phi4:14b` | ~9 GB | Fourth family if you want one. |

For a fabrication sweep you want **at least three families**, not three sizes
of one. Same-family models fail in correlated ways and a "cross-model" result
built on them isn't one.

Pull and check what actually fits:

```bash
ollama pull qwen3:14b
ollama run qwen3:14b "hi"      # loads it
ollama ps                      # SIZE column, and whether it says 100% GPU
```

If `ollama ps` shows anything other than 100% GPU, reduce context or model size.
A partially-offloaded 14B is slower than a fully-resident 8B by a wide margin.

Model tags move. `ollama list` and the library page are the source of truth,
not this file.

---

## 3. The context-window trap

**This is the most important section here.**

Ollama's default context window is small (2048–4096 depending on version). When
your prompt exceeds it, the prompt is **silently truncated**. No error, no
warning, no field in the response.

For this project that failure is catastrophic and nearly invisible. If a model
never sees the second half of a transcript, it correctly refuses to assert a
role that was stated there — and your results record a grounding failure that is
entirely your own bug. It would look like exactly the behaviour you're trying
to measure.

Check your default:

```bash
ollama show qwen3:14b --modelfile | grep -i num_ctx
```

`src/llm.py` guards against this in two ways, and neither should be removed:

1. `num_ctx` is an explicit field on `Call` with no silent default behaviour.
2. `OllamaBackend.complete()` estimates input length and raises
   `ContextOverflow` rather than sending a prompt that would be truncated.

Measure your actual inputs before choosing a value:

```python
from ami import load_meeting
from llm import estimate_tokens

text = "\n".join(u.text for u in load_meeting("ES2002a").utterances)
print(estimate_tokens(text))          # one meeting
# a full four-meeting trial is the number that matters
```

Practical settings on 16 GB with `q8_0` KV cache:

| Model | Workable `num_ctx` |
|---|---|
| `qwen3:8b` | 32k comfortably, 64k tight |
| `qwen3:14b` | 16k comfortably, 32k tight |
| `gemma3:12b` | 16k–32k |

Start at 16k, measure with `ollama ps`, raise until it stops being 100% GPU,
then step back one. Do this once and write the number into your sweep config.

**`OLLAMA_NUM_PARALLEL` splits the context window between concurrent requests.**
Setting it to 4 with `num_ctx=32768` gives each request 8k. Keep it at 1 and get
concurrency by running the sweep's requests sequentially — you have one GPU, so
parallelism buys little anyway.

---

## 4. Structured output

Ollama takes a JSON Schema in the `format` field of `POST /api/chat` and
constrains decoding to it. `src/llm.py` passes `Call.schema` straight through,
and `EXTRACTION_SCHEMA` in that module is the shape to start from: every claim
must carry a verbatim `quote` and a `doc_id`, and there is a first-class
`refusals` array so that declining is a structured output rather than a parse
failure.

Two things this does and does not buy you:

- **Does**: you will always get parseable JSON with the right fields.
- **Does not**: make the content true. A schema-conformant response can contain
  a quote that appears nowhere in the document. That is what `locate()` is for,
  and it is the entire measurement.

---

## 5. Determinism, honestly

Set `temperature=0` and a fixed `seed` — `src/llm.py` does both, and folds
`sample` into the seed so repeated draws of one cell differ in a recorded way.

What this gives you is *reproducibility of the sweep*, not bit-identical
generation. GPU inference varies with batch size and kernel selection, and
quantised models vary more. Treat near-identical as the expectation and log
every response rather than assuming you can regenerate it later.

This is also why the ArkenOS ontology's framing of `Run` is the right one:
identical inputs attempted twice are still two occurrences. Your cache key
includes `sample` for exactly that reason.

---

## 6. If you'd rather use Gemini for part of it

`src/llm.py` has a `GeminiBackend` behind the same interface — `backend_for()`
takes `"gemini:<model>"` and the key comes from `GEMINI_API_KEY` in the
environment, never from argv (command lines are visible to every process and
land in agent transcripts).

Free-tier rate limits and model availability change often; read
<https://ai.google.dev/gemini-api/docs/rate-limits> rather than trusting a
number written down here. For a sweep of thousands of calls you will hit
per-minute limits, so the backoff in the shim matters more than the model choice.

A sensible division: **local models for the sweep** (unlimited, private,
repeatable), **Gemini for one-off reformatting jobs** — turning a messy corpus
into the JSONL shape the harness wants, which is a few hundred calls, not
thousands.

---

## 7. Smoke test before committing to anything

```bash
cd src
python3 llm.py            # locate() + the context guard, no model needed
python3 ontology.py       # stereotypes, derived gates
python3 kg.py             # provenance invariant, path equations
python3 kge.py            # scale check (runs without pykeen)
```

Then one real call:

```python
from llm import Call, backend_for, EXTRACTION_SCHEMA, ground
from ami import load_meeting

m = load_meeting("ES2002a")
corpus = {u.doc_id: t for u, t in m.documents.items()}   # adapt to your loader

r = backend_for("ollama:qwen3:14b").complete(Call(
    model="ollama:qwen3:14b",
    system="Answer only from the transcript. Every claim needs a verbatim quote. "
           "If the transcript does not state something, put it in refusals.",
    user="Who is the project manager?\n\n" + corpus["ES2002a.A"],
    prompt_version="v1",
    schema=EXTRACTION_SCHEMA,
    num_ctx=16384,
))
print(ground(r, corpus))
```

`ground()` splits the response into `grounded` / `unlocated` / `misplaced`.
`unlocated` is the fabrication count. `misplaced` — right words, wrong document —
is its own failure class and worth reporting separately; it is the same mistake
as the ES2002 speaker-letter trap, one level up.
