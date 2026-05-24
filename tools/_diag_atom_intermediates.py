"""Run ATOM Step-3.5-Flash on a single prompt with intermediate capture.

Patches are installed inside every spawned worker via the sitecustomize.py
module under tools/diag_step3p5_pp/. After the engine shuts down each
worker writes its captured intermediates to
$ATOM_DIAG_STEP3P5_OUT.<pid>.json. This script aggregates all of them.
"""

import glob
import json
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
# Activate the sitecustomize-based capture in this process and every child.
os.environ["ATOM_DIAG_STEP3P5"] = "1"
DEFAULT_OUT_BASE = "/tmp/atom_diag_step3p5"
os.environ.setdefault("ATOM_DIAG_STEP3P5_OUT", DEFAULT_OUT_BASE)

# Make sure tools/diag_step3p5_pp is on PYTHONPATH for spawned workers.
SITECUSTOMIZE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diag_step3p5_pp"
)
existing = os.environ.get("PYTHONPATH", "")
if SITECUSTOMIZE_DIR not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [SITECUSTOMIZE_DIR] + ([existing] if existing else [])
    )

# Also load it into the parent process by importing now.
if SITECUSTOMIZE_DIR not in sys.path:
    sys.path.insert(0, SITECUSTOMIZE_DIR)
import sitecustomize  # noqa: F401  - triggers patch registration here too

from atom import SamplingParams  # noqa: E402
from atom.model_engine.arg_utils import EngineArgs  # noqa: E402

PROMPT = "What is 2+2?"
MODEL = "/data/jamesmit/models/Step-3.5-Flash-FP8"
OUT_BASE = os.environ["ATOM_DIAG_STEP3P5_OUT"]
FINAL_OUT = os.environ.get("ATOM_DIAG_STEP3P5_FINAL", "/tmp/atom_diag_step3p5_merged.json")


def _clean_old_dumps():
    for path in glob.glob(f"{OUT_BASE}.*.json"):
        try:
            os.unlink(path)
        except OSError:
            pass


def _merge_dumps():
    merged = {"prompt": PROMPT, "model": MODEL, "workers": {}}
    for path in sorted(glob.glob(f"{OUT_BASE}.*.json")):
        pid = path.rsplit(".", 2)[1]
        with open(path) as handle:
            try:
                merged["workers"][pid] = json.load(handle)
            except json.JSONDecodeError as exc:
                merged["workers"][pid] = {"error": f"decode failed: {exc!r}"}
    return merged


def _wait_for_dumps(timeout=30.0, min_count=1, stable_seconds=2.0):
    """Poll until at least `min_count` worker dumps exist and stop changing."""
    import time

    deadline = time.time() + timeout
    last_count = -1
    last_change = time.time()
    while time.time() < deadline:
        files = sorted(glob.glob(f"{OUT_BASE}.*.json"))
        if len(files) != last_count:
            last_count = len(files)
            last_change = time.time()
        if last_count >= min_count and (time.time() - last_change) >= stable_seconds:
            return files
        time.sleep(0.25)
    return sorted(glob.glob(f"{OUT_BASE}.*.json"))


def main():
    _clean_old_dumps()
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

    _wait_for_dumps()
    merged = _merge_dumps()
    merged["output_text"] = outputs[0]["text"] if outputs else ""

    with open(FINAL_OUT, "w") as handle:
        json.dump(merged, handle, indent=2)

    summary = {
        "engine": "atom",
        "prompt": PROMPT,
        "output_text": merged["output_text"],
        "worker_pids": list(merged["workers"].keys()),
        "worker_capture_keys": {
            pid: sorted(data.keys())[:8] for pid, data in merged["workers"].items()
        },
        "merged_path": FINAL_OUT,
    }
    print("__DIAG_JSON_BEGIN__")
    print(json.dumps(summary, indent=2))
    print("__DIAG_JSON_END__")


if __name__ == "__main__":
    main()
