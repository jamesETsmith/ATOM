"""Launcher: bf16 GemmaRMSNorm patch via sitecustomize on every spawned worker.

Set PYTHONPATH so that `tools/diag_step3p5_bf16_pp/sitecustomize.py` is loaded
in every Python process (parent, EngineCore, ModelRunner workers). The
sitecustomize then patches GemmaRMSNorm.forward_static and the Triton
norm+RoPE+cache kernel to use SGLang-style bf16 (weight + 1.0) semantics.

Usage:
  PYTHONPATH=tools/diag_step3p5_bf16_pp \
  ATOM_DIAG_BF16=1 ATOM_DIAG_BF16_DEBUG=1 \
  python tools/run_step3p5_bf16_diag.py

The script self-execs into the right environment if those env vars aren't
already set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from glob import glob


HERE = os.path.dirname(os.path.abspath(__file__))
SITE_DIR = os.path.join(HERE, "diag_step3p5_bf16_pp")
OUT_BASE = "/tmp/atom_diag_bf16"


def _self_exec_with_env():
    """Re-exec ourselves with PYTHONPATH and ATOM_DIAG_BF16 set."""
    env = os.environ.copy()
    env["PYTHONPATH"] = SITE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["ATOM_DIAG_BF16"] = "1"
    env.setdefault("ATOM_DIAG_BF16_DEBUG", "1")
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    env.setdefault("AITER_LOG_LEVEL", "WARNING")
    # Clear stale per-pid dumps.
    for f in glob(f"{OUT_BASE}.*.json"):
        try:
            os.remove(f)
        except OSError:
            pass
    env["_BF16_DIAG_BOOTSTRAPPED"] = "1"
    sys.stdout.flush()
    os.execvpe(sys.executable, [sys.executable, __file__] + sys.argv[1:], env)


if os.environ.get("_BF16_DIAG_BOOTSTRAPPED") != "1":
    _self_exec_with_env()


# -------- Now we're in the bootstrapped environment --------

from atom import SamplingParams  # noqa: E402
from atom.model_engine.arg_utils import EngineArgs  # noqa: E402

PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
    "Translate to French: Hello, world.",
]
MODEL = "/data/jamesmit/models/Step-3.5-Flash-FP8"


def main():
    print(f"[run] PYTHONPATH={os.environ.get('PYTHONPATH','')}")
    print(f"[run] ATOM_DIAG_BF16={os.environ.get('ATOM_DIAG_BF16','')}")
    sys.stdout.flush()

    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=1,
        enforce_eager=True,
        kv_cache_dtype="fp8",
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        cudagraph_capture_sizes="[1]",
        level=0,
    )
    llm = engine_args.create_engine()
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=30))
    llm.close()
    results = [
        {"prompt": p, "completion": o["text"]} for p, o in zip(PROMPTS, outputs)
    ]

    # Wait briefly for spawned workers to flush their atexit dumps.
    import time
    time.sleep(2)

    counters = []
    for path in sorted(glob(f"{OUT_BASE}.*.json")):
        with open(path) as f:
            counters.append(json.load(f))

    print("__SUMMARY_BEGIN__")
    print(json.dumps({
        "engine": "atom-bf16-via-sitecustomize",
        "results": results,
        "per_pid_counters": counters,
        "total_processes": len(counters),
    }, indent=2))
    print("__SUMMARY_END__")


if __name__ == "__main__":
    main()
