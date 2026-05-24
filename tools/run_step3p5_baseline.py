"""Minimal baseline test: unpatched ATOM, same prompts. For A/B comparison."""
from __future__ import annotations
import json, os, sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

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
    print("__SUMMARY_BEGIN__")
    print(json.dumps({"engine": "atom-baseline", "results": results}, indent=2))
    print("__SUMMARY_END__")

if __name__ == "__main__":
    main()
