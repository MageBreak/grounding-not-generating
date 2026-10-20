#!/usr/bin/env python3
"""What is actually present on this machine, and what has actually been run.

Read-only. Touches nothing, installs nothing, and deliberately never opens the
gold-label file -- it checks that it exists and that it is *not* inside the
workspace, which is the only thing worth knowing about it from here.

    python3 scripts/status.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, NO, WARN = "  ok  ", " MISS ", " warn "


def line(tag: str, label: str, detail: str = "") -> None:
    print(f"[{tag}] {label:<34} {detail}")


def head(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def sh(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                           # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
def check_repo() -> None:
    head("1. REPOSITORY")
    line(OK if (ROOT / ".git").is_dir() else NO, "git repo", str(ROOT))

    branch = sh(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    commit = sh(["git", "-C", str(ROOT), "log", "-1", "--format=%h %s"])
    dirty = sh(["git", "-C", str(ROOT), "status", "--short"])
    if branch:
        line(OK, "branch", branch)
        line(OK, "last commit", commit or "?")
        if dirty:
            line(WARN, "uncommitted changes", f"{len(dirty.splitlines())} paths")
            for p in dirty.splitlines()[:8]:
                print(f"         {p}")
        else:
            line(OK, "working tree", "clean")

    expected = {
        "src/ami.py": "NITE resolver",
        "src/schema.py": "gates",
        "src/prototype.py": "regex baseline experiment",
        "src/run.py": "corpus tour",
        "src/llm.py": "backend shim + locate()",
        "src/ontology.py": "UFO stereotypes, derived gates",
        "src/kg.py": "provenance-mandatory graph",
        "src/kge.py": "embeddings, suggest-only",
        "docs/LOCAL_MODEL.md": "",
        "docs/DATASETS.md": "",
        "docs/pipeline.html": "",
    }
    print()
    missing = []
    for rel, why in expected.items():
        p = ROOT / rel
        if p.exists():
            line(OK, rel, f"{p.stat().st_size:>7,} B  {why}")
        else:
            line(NO, rel, why)
            missing.append(rel)
    if missing:
        print(f"\n  -> {len(missing)} file(s) absent. If this is a fresh clone of an")
        print("     older commit, the four newer modules will not be here yet.")


def check_corpus() -> None:
    head("2. AMI CORPUS")
    data = Path(os.environ.get("AMI_ROOT", ROOT / "data" / "ami"))
    if not data.exists():
        line(NO, "data/ami", f"{data} does not exist")
        print("\n  -> bash scripts/fetch_ami.sh        (~22 MB from Edinburgh)")
        return
    line(OK, "data/ami", str(data))

    for sub, need in (("words", True), ("segments", True), ("dialogueActs", False),
                      ("extractive", False), ("abstractive", False),
                      ("corpusResources", False), ("ontologies", False)):
        d = data / sub
        if d.is_dir():
            n = len(list(d.glob("*.xml")))
            line(OK, f"  {sub}/", f"{n} xml")
        else:
            line(NO if need else WARN, f"  {sub}/", "absent")

    words = data / "words"
    if words.is_dir():
        trials = sorted({f.name.split(".")[0] for f in words.glob("*.xml")})
        line(OK, "distinct meetings", str(len(trials)))
        line(OK, "sample", ", ".join(trials[:6]) + (" ..." if len(trials) > 6 else ""))


def check_wall() -> None:
    head("3. THE GOLD-LABEL WALL")
    inside = list((ROOT / "data").rglob("meetings.xml")) if (ROOT / "data").exists() else []
    if inside:
        line(WARN, "gold inside the workspace", str(inside[0]))
        print("\n  -> any agent browsing data/ can read it. Move it out:")
        print("     mkdir -p ~/gold && mv <that path> ~/gold/")
    else:
        line(OK, "no meetings.xml under the repo", "")

    gold = Path.home() / "gold"
    if gold.is_dir():
        names = [f.name for f in gold.iterdir() if f.is_file()]
        line(OK, "~/gold exists", ", ".join(names) if names else "(empty)")
    else:
        line(NO, "~/gold", "absent -- scoring will have nothing to score against")

    # standing instructions
    agents = ROOT / ".optimus" / "agents"
    if agents.is_dir():
        empty = [d.name for d in agents.iterdir()
                 if (d / "AGENTS.md").exists() and (d / "AGENTS.md").stat().st_size == 0]
        total = len([d for d in agents.iterdir() if d.is_dir()])
        if empty:
            line(WARN, "AGENTS.md empty for", ", ".join(empty))
        else:
            line(OK, "AGENTS.md written", f"{total} agents")
    else:
        line(WARN, ".optimus/agents", "absent -- not an ArkenOS workspace, or new")


def check_runs() -> None:
    head("4. WHAT HAS ACTUALLY BEEN RUN")
    candidates = []
    for pat in ("runs/*.jsonl", "out/*.jsonl", "*.jsonl", "results/*.jsonl"):
        candidates += list(ROOT.glob(pat))
    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        line(NO, "run store", "no .jsonl output anywhere in the repo")
        print("\n  -> nothing has been swept yet. The numbers in docs/RESULTS.md")
        print("     came from prototype.py (regex), not from a model.")
        return

    for p in sorted(candidates):
        rows, models, prompts, errs = 0, set(), set(), 0
        try:
            with p.open() as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    rows += 1
                    try:
                        d = json.loads(raw)
                    except Exception:                           # noqa: BLE001
                        continue
                    if isinstance(d, dict):
                        models.add(str(d.get("model", "?")))
                        prompts.add(str(d.get("prompt_version", "?")))
                        if d.get("ok") is False or d.get("error"):
                            errs += 1
        except Exception as e:                                  # noqa: BLE001
            line(WARN, p.name, f"unreadable: {e}")
            continue
        line(OK, p.relative_to(ROOT).as_posix(), f"{rows:,} rows")
        print(f"         models:  {', '.join(sorted(models)) or '-'}")
        print(f"         prompts: {', '.join(sorted(prompts)) or '-'}")
        if errs:
            print(f"         errors:  {errs}")


def check_python() -> None:
    head("5. PYTHON AND DEPENDENCIES")
    line(OK, "python", sys.version.split()[0])
    for mod, why in (("ollama", "optional -- llm.py uses urllib, not this"),
                     ("google.genai", "needed only for the Gemini backend"),
                     ("pykeen", "needed only for kge.train()"),
                     ("torch", "needed only for kge"),
                     ("networkx", "needed only for kg.to_networkx()")):
        try:
            __import__(mod)
            line(OK, mod, why)
        except ImportError:
            line(WARN, mod, f"absent -- {why}")


def check_ollama() -> None:
    head("6. LOCAL INFERENCE")
    line(OK if shutil.which("ollama") else NO, "ollama binary",
         shutil.which("ollama") or "not on PATH -- curl -fsSL https://ollama.com/install.sh | sh")

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=4) as r:
            tags = json.loads(r.read())
        line(OK, "server reachable", host)
        models = tags.get("models", [])
        if not models:
            line(WARN, "models pulled", "none -- ollama pull qwen3:14b")
        for m in models:
            gb = m.get("size", 0) / 1e9
            line(OK, f"  {m.get('name', '?')}", f"{gb:.1f} GB")
    except Exception:                                           # noqa: BLE001
        line(NO, "server reachable", f"{host} -- start it with: ollama serve")

    smi = sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
              "--format=csv,noheader"])
    line(OK if smi else WARN, "gpu", smi or "nvidia-smi unavailable")

    for var in ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE",
                "OLLAMA_KEEP_ALIVE", "OLLAMA_NUM_PARALLEL"):
        v = os.environ.get(var)
        line(OK if v else WARN, f"  ${var}", v or "unset (see docs/LOCAL_MODEL.md)")


def check_selftests() -> None:
    head("7. SELF-TESTS")
    src = ROOT / "src"
    for name in ("llm.py", "ontology.py", "kg.py", "kge.py"):
        p = src / name
        if not p.exists():
            line(NO, name, "file absent")
            continue
        try:
            r = subprocess.run([sys.executable, name], cwd=src,
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                line(OK, name, "passes")
            else:
                line(WARN, name, (r.stderr.strip().splitlines() or ["failed"])[-1][:60])
        except Exception as e:                                  # noqa: BLE001
            line(WARN, name, f"{type(e).__name__}: {e}")


def verdict() -> None:
    head("NEXT")
    steps: list[str] = []
    if not (ROOT / "src" / "llm.py").exists():
        steps.append("git pull -- the newer modules are not in this checkout")
    if not Path(os.environ.get("AMI_ROOT", ROOT / "data" / "ami")).exists():
        steps.append("bash scripts/fetch_ami.sh")
    if not (Path.home() / "gold").is_dir():
        steps.append("derive ~/gold/ from corpusResources, then move it out of the repo")
    if not shutil.which("ollama"):
        steps.append("install ollama, then: ollama pull qwen3:14b gemma3:12b")
    if not steps:
        steps.append("everything present -- build the runner and start the sweep")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    print()


if __name__ == "__main__":
    print(f"grounding-not-generating :: status :: {ROOT}")
    check_repo()
    check_corpus()
    check_wall()
    check_runs()
    check_python()
    check_ollama()
    check_selftests()
    verdict()
