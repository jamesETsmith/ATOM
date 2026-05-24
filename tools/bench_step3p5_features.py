"""Benchmark Step-3.5-Flash-FP8: baseline, EP, MTP, and EP+MTP.

Launches one server per configuration, runs warmup + measurement, kills server,
moves on. Writes JSONL to stdout and a summary table at the end.

Usage:
    .venv-gfx942/bin/python tools/bench_step3p5_features.py

Override configs with TP_CONFIGS env var (default runs all feature combos).
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
CONCURRENCIES = [1, 8]

# Each entry: (tp, enable_expert_parallel, enable_mtp, label)
CONFIGS = [
    (1, False, False, "TP=1 baseline"),
    (4, True, False, "TP=4 EP"),
    (8, True, False, "TP=8 EP"),
    (1, False, True, "TP=1 MTP"),
    (4, True, True, "TP=4 EP+MTP"),
    (8, True, True, "TP=8 EP+MTP"),
]

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


def launch_server(tp: int, ep: bool, mtp: bool) -> subprocess.Popen:
    env = server_env()
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
    if mtp:
        cmd.extend([
            "--method", "mtp",
            "--num-speculative-tokens", "1",
        ])
    suffix = f"tp{tp}"
    if ep:
        suffix += "_ep"
    if mtp:
        suffix += "_mtp"
    log_path = REPO / f"bench_{suffix}.log"
    log_f = open(log_path, "wb")
    print(f"[bench] launching {suffix}, log -> {log_path}", flush=True)
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(REPO),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_ready(proc: subprocess.Popen, timeout: float = 1500.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if r.status_code == 200:
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
        pass


def make_prompt(n_tokens: int) -> str:
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


def bench_config(tp: int, ep: bool, mtp: bool, label: str) -> list[dict]:
    rows = []
    proc = launch_server(tp, ep, mtp)
    try:
        ok = wait_ready(proc)
        if not ok:
            print(f"[bench] {label} server failed to come up", flush=True)
            # Dump last 50 lines of log for diagnosis
            suffix = f"tp{tp}"
            if ep:
                suffix += "_ep"
            if mtp:
                suffix += "_mtp"
            log_path = REPO / f"bench_{suffix}.log"
            if log_path.exists():
                lines = log_path.read_text(errors="replace").splitlines()
                for line in lines[-50:]:
                    print(f"  LOG: {line}", flush=True)
            return rows
        prompt = make_prompt(ISL)
        # Warmup
        print(f"[bench] {label} warming up (c=1, 64 tokens)...", flush=True)
        run_concurrent(prompt, 64, 1)
        print(f"[bench] {label} warming up (c=8, 64 tokens)...", flush=True)
        run_concurrent(prompt, 64, 8)
        for c in CONCURRENCIES:
            print(
                f"[bench] {label} measuring ISL={ISL} OSL={OSL} c={c}...",
                flush=True,
            )
            t0 = time.time()
            results = run_concurrent(prompt, OSL, c)
            wall = time.time() - t0
            summary = summarize(results, wall)
            row = {
                "label": label,
                "tp": tp,
                "ep": ep,
                "mtp": mtp,
                "isl": ISL,
                "osl": OSL,
                "concurrency": c,
                **summary,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        kill_server(proc)
        time.sleep(5)
    return rows


def main() -> None:
    all_rows = []
    for tp, ep, mtp, label in CONFIGS:
        all_rows.extend(bench_config(tp, ep, mtp, label))

    out_path = REPO / "bench_step3p5_features_results.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n[bench] wrote {out_path}", flush=True)

    # Pretty table
    print()
    hdr = (
        f"{'Config':<18} {'C':>3} {'TTFT(ms)':>10} {'TPOT(ms)':>10} "
        f"{'dec tok/s':>10} {'out tok/s':>12} {'wall(s)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(
            f"{r['label']:<18} {r['concurrency']:>3} "
            f"{r['avg_ttft_ms']:>10.1f} {r['avg_tpot_ms']:>10.2f} "
            f"{r['decode_tok_per_s_per_req']:>10.1f} "
            f"{r['output_tok_per_s_aggregate']:>12.1f} "
            f"{r['wall_s']:>9.1f}"
        )

    # Comparison with previous results
    prev_path = REPO / "bench_step3p5_tp_results.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())
        print("\n\n=== Comparison with previous results ===")
        print(f"{'Config':<18} {'C':>3} {'prev tok/s':>12} {'new tok/s':>12} {'delta':>8}")
        print("-" * 60)
        for r in all_rows:
            for p in prev:
                if p["tp"] == r["tp"] and p.get("ep") == r["ep"] and p["concurrency"] == r["concurrency"]:
                    prev_toks = p["decode_tok_per_s_per_req"]
                    new_toks = r["decode_tok_per_s_per_req"]
                    delta = ((new_toks - prev_toks) / prev_toks * 100) if prev_toks else 0
                    print(
                        f"{r['label']:<18} {r['concurrency']:>3} "
                        f"{prev_toks:>12.1f} {new_toks:>12.1f} "
                        f"{delta:>+7.1f}%"
                    )


if __name__ == "__main__":
    sys.exit(main() or 0)
