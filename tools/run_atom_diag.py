"""Driver script for Step-3.5 ATOM diagnostic.

The actual instrumentation lives in `sitecustomize.py` placed on PYTHONPATH,
so it runs in the spawn worker process (where the model actually executes).
This driver only creates the engine and triggers a single short generation.
"""

import json
import os
import sys

import torch  # noqa: F401  (ensure torch loads in main pid too)
from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs

PROMPT = os.environ.get("STEP3P5_DIAG_PROMPT", "What is 2+2?")
MODEL = "/data/jamesmit/models/Step-3.5-Flash-FP8"


def main() -> None:
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
    outputs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1))
    llm.close()
    print("__DIAG_DRIVER_BEGIN__")
    print(json.dumps({
        "prompt": PROMPT,
        "outputs_repr": repr(outputs),
        "outputs_keys": list(outputs[0].keys()) if outputs else [],
    }, indent=2, default=str))
    print("__DIAG_DRIVER_END__")
    print(f"intermediates flushed by worker to {os.environ.get('STEP3P5_DIAG_OUT', '/tmp/step3p5_diag')}/", file=sys.stderr)


if __name__ == "__main__":
    main()
