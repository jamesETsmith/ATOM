"""Runs inside the rocm/sgl-dev container.

Assumes:
  PYTHONPATH includes /diag_sglang_pp (mounted from the host) so the
  sitecustomize.py in that directory is auto-loaded by every Python
  process started from this script (including SGLang's scheduler/worker
  subprocesses, since they inherit the env).

Calls SGLang's Engine() with the same FP8 mixed-precision config we use
for serving Step-3.5-Flash and writes a per-PID dump for every process
that gets the patches. The host-side script then aggregates them.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("SGLANG_SET_CPU_AFFINITY", "0")
os.environ.setdefault("SGLANG_DIAG", "1")
os.environ.setdefault("SGLANG_DIAG_OUT", "/output/sglang_diag")
os.environ.setdefault("SGLANG_DIAG_DEBUG", "1")

PROMPT = "What is 2+2?"
MODEL = os.environ.get("SGLANG_MODEL", "/models/Step-3.5-Flash-FP8")

# Force-import sitecustomize so the parent process also patches before
# spinning up the engine. (Workers inherit PYTHONPATH and trigger their
# own auto-load.)
import sitecustomize  # noqa: F401
from sglang import Engine  # noqa: E402


def main() -> None:
    engine = Engine(
        model_path=MODEL,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        attention_backend="triton",
        disable_cuda_graph=True,
        log_level="warning",
        max_total_tokens=16384,
        trust_remote_code=True,
    )
    out = engine.generate(
        [PROMPT],
        sampling_params={"temperature": 0.0, "max_new_tokens": 1, "top_p": 1.0},
    )
    text = out[0].get("text", "") if out else ""
    engine.shutdown()

    summary = {
        "engine": "sglang",
        "prompt": PROMPT,
        "model": MODEL,
        "output_text": text,
    }
    summary_path = os.environ["SGLANG_DIAG_OUT"] + ".summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as h:
        json.dump(summary, h, indent=2)

    print("__DIAG_JSON_BEGIN__")
    print(json.dumps(summary, indent=2))
    print("__DIAG_JSON_END__")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
