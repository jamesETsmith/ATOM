"""Inline launcher: bf16 diag with sitecustomize on all workers.

Run directly with env set on the command line:
  PYTHONPATH=tools/diag_step3p5_bf16_pp ATOM_DIAG_BF16=1 ATOM_DIAG_BF16_DEBUG=1 \
  HIP_VISIBLE_DEVICES=0 AITER_LOG_LEVEL=WARNING \
  .venv-gfx942/bin/python tools/run_step3p5_bf16_v2.py
"""
from __future__ import annotations
import json, os, sys, time
from glob import glob

OUT_BASE = "/tmp/atom_diag_bf16"
for f in glob(f"{OUT_BASE}.*.json"):
    try: os.remove(f)
    except: pass

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs

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
    results = [{"prompt": p, "completion": o["text"]} for p, o in zip(PROMPTS, outputs)]

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
