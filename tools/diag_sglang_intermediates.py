"""Host-side wrapper to run SGLang Step-3.5-Flash intermediate capture.

Launches the rocm/sgl-dev container with:
  - sglang_patches/{step3p5.py,fp8.py} mounted over the in-image SGLang
    files so the FP8 mixed-precision Step-3.5 checkpoint loads.
  - tools/diag_sglang_pp/ mounted at /diag_sglang_pp and on PYTHONPATH so
    sitecustomize.py auto-installs in every Python process the container
    spawns (including SGLang's scheduler/worker subprocesses).
  - /tmp/sglang_diag_out mounted at /output so per-PID dumps survive
    container shutdown.

After the container exits, aggregates per-PID dumps into a single JSON
that mirrors the shape of /tmp/atom_diag_step3p5_merged.json so the two
can be diffed directly.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HOST_OUT = pathlib.Path("/tmp/sglang_diag_out")
HOST_OUT.mkdir(parents=True, exist_ok=True)
for p in HOST_OUT.glob("sglang_diag.*.json"):
    p.unlink()
summary = HOST_OUT / "sglang_diag.summary.json"
if summary.exists():
    summary.unlink()

MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
MODEL_CT = "/models/Step-3.5-Flash-FP8"
SGL_IMAGE = "rocm/sgl-dev:v0.5.10.post1-rocm720-mi30x-20260503"
CTNAME = "sgl_diag_step3p5"


def host_cpu_count() -> int:
    try:
        with open("/proc/cpuinfo") as f:
            return sum(1 for line in f if line.startswith("processor"))
    except OSError:
        return os.cpu_count() or 1


def kill_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", CTNAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run() -> int:
    kill_container()
    cmd = [
        "docker", "run", "--rm",
        "--name", CTNAME,
        "--network=host",
        "--ipc=host",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", "32g",
        "--cpuset-cpus", f"0-{host_cpu_count() - 1}",
        "-e", "HIP_VISIBLE_DEVICES=0",
        "-e", "SGLANG_USE_AITER=1",
        "-e", "SGLANG_SET_CPU_AFFINITY=0",
        "-e", "SGLANG_DIAG=1",
        "-e", "SGLANG_DIAG_DEBUG=1",
        "-e", "SGLANG_DIAG_OUT=/output/sglang_diag",
        "-e", "PYTHONPATH=/diag_sglang_pp",
        # Model
        "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
        # SGLang patches required for FP8 mixed-precision Step-3.5
        "-v", f"{REPO}/tools/sglang_patches/step3p5.py:/sgl-workspace/sglang/python/sglang/srt/models/step3p5.py:ro",
        "-v", f"{REPO}/tools/sglang_patches/fp8.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py:ro",
        # sitecustomize-based capture infrastructure
        "-v", f"{REPO}/tools/diag_sglang_pp:/diag_sglang_pp:ro",
        # Output directory
        "-v", f"{HOST_OUT}:/output",
        SGL_IMAGE,
        "python3", "/diag_sglang_pp/run_inside.py",
    ]
    print("[diag-sglang] launching container...", flush=True)
    print(" ".join(cmd), flush=True)
    res = subprocess.run(cmd)
    return res.returncode


def aggregate() -> dict:
    merged = {
        "engine": "sglang",
        "prompt": "What is 2+2?",
        "model": MODEL_CT,
        "workers": {},
    }
    files = sorted(HOST_OUT.glob("sglang_diag.*.json"))
    for path in files:
        if path.name.endswith(".summary.json"):
            continue
        # filename pattern: sglang_diag.<pid>.json
        parts = path.name.split(".")
        pid = parts[1] if len(parts) >= 3 else path.stem
        try:
            merged["workers"][pid] = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            merged["workers"][pid] = {"error": f"decode failed: {exc!r}"}
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            merged["output_text"] = data.get("output_text", "")
        except json.JSONDecodeError:
            pass
    return merged


def main() -> int:
    rc = run()
    if rc != 0:
        print(f"[diag-sglang] container exited rc={rc}", file=sys.stderr)
    merged = aggregate()
    out_path = "/tmp/sglang_diag_step3p5_merged.json"
    with open(out_path, "w") as h:
        json.dump(merged, h, indent=2)
    summary_out = {
        "engine": "sglang",
        "prompt": merged["prompt"],
        "output_text": merged.get("output_text", ""),
        "worker_pids": list(merged["workers"].keys()),
        "worker_capture_keys": {
            pid: sorted(data.keys())[:8] for pid, data in merged["workers"].items()
        },
        "merged_path": out_path,
    }
    print("__DIAG_JSON_BEGIN__")
    print(json.dumps(summary_out, indent=2))
    print("__DIAG_JSON_END__")
    return rc


if __name__ == "__main__":
    sys.exit(main())
