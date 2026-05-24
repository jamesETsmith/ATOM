"""Benchmark Step-3.5-Flash-FP8 across TP=1,2,4,8 at ISL=1024 OSL=8192.

Launches one server per TP value, runs warmup + measurement, kills server,
moves on. Writes JSONL to stdout and a summary table at the end.
"""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MODEL = "/data/jamesmit/models/Step-3.5-Flash-FP8"
PORT = 8000
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}/v1/completions"
ISL = 1024
OSL = 8192
# Each entry: (tp, enable_expert_parallel)
# moe_intermediate_size=1280 is only divisible by FP8 block_n=128 at TP in {1,2,5,10};
# for TP=4 and TP=8 we must shard MoE via expert parallelism instead.
_DEFAULT_CONFIGS = "1:0,2:0,4:1,8:1"
TP_CONFIGS = []
for item in os.environ.get("TP_CONFIGS", _DEFAULT_CONFIGS).split(","):
    tp_str, ep_str = item.split(":")
    TP_CONFIGS.append((int(tp_str), bool(int(ep_str))))
CONCURRENCIES = [1, 8]  # per TP

REPO = Path("/home/AMD/jamesmit/apps/ATOM")
VENV = REPO / ".venv-gfx942"
VENV_CORE = VENV / "lib/python3.12/site-packages/_rocm_sdk_core"
VENV_DEVEL = VENV / "lib/python3.12/site-packages/_rocm_sdk_devel"
VENV_LIBS = VENV / "lib/python3.12/site-packages/_rocm_sdk_libraries_gfx94X_dcgpu/lib"


def server_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{VENV/'bin'}:{VENV_CORE/'bin'}:{VENV_DEVEL/'bin'}:{env.get('PATH','')}"
    env["ROCM_HOME"] = str(VENV_DEVEL)
    env["ROCM_PATH"] = str(VENV_DEVEL)
    env["HIP_PATH"] = str(VENV_CORE)
    env["HIP_DEVICE_LIB_PATH"] = str(VENV_CORE / "lib/llvm/amdgcn/bitcode")
    env["LD_LIBRARY_PATH"] = (
        f"{VENV_LIBS}:{VENV_CORE/'lib'}:{VENV_DEVEL/'lib'}:{env.get('LD_LIBRARY_PATH','')}"
    )
    env["LIBRARY_PATH"] = (
        f"{VENV_CORE/'lib'}:{VENV_DEVEL/'lib'}:{VENV_LIBS}:{env.get('LIBRARY_PATH','')}"
    )
    env["CPLUS_INCLUDE_PATH"] = (
        f"{VENV_DEVEL/'include'}:{VENV_CORE/'include'}:{env.get('CPLUS_INCLUDE_PATH','')}"
    )
    env["HF_HOME"] = "/data/jamesmit/hf_cache"
    env["HF_HUB_CACHE"] = "/data/jamesmit/hf_cache"
    env["TRANSFORMERS_CACHE"] = "/data/jamesmit/hf_cache"
    env["TRITON_CACHE_DIR"] = "/data/jamesmit/triton_cache"
    env["AITER_LOG_LEVEL"] = "WARNING"
    return env


def launch_server(tp: int, ep: bool) -> subprocess.Popen:
    env = server_env()
    # Pin to first `tp` GPUs
    env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp))
    cmd = [
        str(VENV / "bin/python"),
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        MODEL,
        "--kv_cache_dtype",
        "fp8",
        "-tp",
        str(tp),
        "--gpu-memory-utilization",
        "0.92",
        "--port",
        str(PORT),
    ]
    if ep:
        cmd.append("--enable-expert-parallel")
    suffix = f"tp{tp}" + ("_ep" if ep else "")
    log_path = REPO / f"bench_{suffix}.log"
    log_f = open(log_path, "wb")
    print(f"[bench] launching TP={tp}, log -> {log_path}", flush=True)
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(REPO),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_ready(proc: subprocess.Popen, timeout: float = 1500.0) -> bool:
    """Poll /health until 200 or timeout, while checking proc didn't die."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if r.status_code == 200:
                # Also confirm the server can list models (model loaded)
                m = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=2)
                if m.status_code == 200 and MODEL in m.text:
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def kill_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Reap any orphan multiprocessing.spawn helpers that still hold VRAM.
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "multiprocessing.spawn"], text=True
        ).strip()
        if out:
            for pid in out.splitlines():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
    except subprocess.CalledProcessError:
        pass  # nothing to reap


def make_prompt(n_tokens: int) -> str:
    # Roughly 1 word per BPE token; pad with simple tokens.
    return " ".join(["the"] * n_tokens)


def one_request(prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=1200)
    elapsed = time.time() - t0
    r.raise_for_status()
    j = r.json()
    usage = j["usage"]
    return {
        "ttft_ms": usage.get("ttft_s", 0.0) * 1000.0,
        "tpot_ms": usage.get("tpot_s", 0.0) * 1000.0,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "wall_s": elapsed,
    }


def run_concurrent(prompt: str, max_tokens: int, concurrency: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, prompt, max_tokens) for _ in range(concurrency)]
        return [f.result() for f in as_completed(futs)]


def summarize(results: list[dict], wall_s: float) -> dict:
    n = len(results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / n
    avg_tpot = sum(r["tpot_ms"] for r in results) / n
    total_in = sum(r["prompt_tokens"] for r in results)
    total_out = sum(r["completion_tokens"] for r in results)
    decode_toks = (1000.0 / avg_tpot) if avg_tpot > 0 else 0.0
    output_toks = total_out / wall_s
    total_toks = (total_in + total_out) / wall_s
    return {
        "n": n,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "decode_tok_per_s_per_req": decode_toks,
        "output_tok_per_s_aggregate": output_toks,
        "total_tok_per_s_aggregate": total_toks,
        "wall_s": wall_s,
        "total_in": total_in,
        "total_out": total_out,
    }


def bench_tp(tp: int, ep: bool) -> list[dict]:
    label = f"TP={tp}" + (" (EP)" if ep else "")
    rows = []
    proc = launch_server(tp, ep)
    try:
        ok = wait_ready(proc)
        if not ok:
            print(f"[bench] {label} server failed to come up", flush=True)
            return rows
        prompt = make_prompt(ISL)
        # Warmup: trigger any first-time JIT / cudagraph capture
        print(f"[bench] {label} warming up (concurrency=1, max_tokens=64)...", flush=True)
        run_concurrent(prompt, 64, 1)
        # Second warmup at higher concurrency to capture larger graphs
        print(f"[bench] {label} warming up (concurrency=8, max_tokens=64)...", flush=True)
        run_concurrent(prompt, 64, 8)
        for c in CONCURRENCIES:
            print(
                f"[bench] {label} measuring ISL={ISL} OSL={OSL} concurrency={c}...",
                flush=True,
            )
            t0 = time.time()
            results = run_concurrent(prompt, OSL, c)
            wall = time.time() - t0
            summary = summarize(results, wall)
            row = {"tp": tp, "ep": ep, "isl": ISL, "osl": OSL, "concurrency": c, **summary}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        kill_server(proc)
        # Brief cooldown
        time.sleep(5)
    return rows


def main() -> None:
    all_rows = []
    for tp, ep in TP_CONFIGS:
        all_rows.extend(bench_tp(tp, ep))

    out_path = REPO / "bench_step3p5_tp_results.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n[bench] wrote {out_path}", flush=True)

    # Pretty table
    print()
    hdr = (
        f"{'TP':>3} {'EP':>3} {'C':>3} {'TTFT(ms)':>10} {'TPOT(ms)':>10} "
        f"{'dec tok/s/req':>14} {'out tok/s':>12} {'tot tok/s':>12} {'wall(s)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(
            f"{r['tp']:>3} {('Y' if r.get('ep') else 'N'):>3} {r['concurrency']:>3} "
            f"{r['avg_ttft_ms']:>10.1f} {r['avg_tpot_ms']:>10.2f} "
            f"{r['decode_tok_per_s_per_req']:>14.1f} "
            f"{r['output_tok_per_s_aggregate']:>12.1f} "
            f"{r['total_tok_per_s_aggregate']:>12.1f} "
            f"{r['wall_s']:>9.1f}"
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
