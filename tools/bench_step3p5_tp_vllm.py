"""Benchmark Step-3.5-Flash-FP8 on vLLM (rocm/vllm image) at TP=1,2,4,8.

Mirrors tools/bench_step3p5_tp.py but launches vLLM inside a docker container
instead of ATOM in a venv. Same model, ISL, OSL, and concurrencies so the
results are directly comparable.

TTFT is measured via /v1/completions streaming (time-to-first-chunk).
TPOT is computed as (wall - ttft) / (completion_tokens - 1).
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

VLLM_IMAGE = (
    "rocm/vllm:rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"
)
MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
MODEL_CT = "/models/Step-3.5-Flash-FP8"
HF_CACHE_HOST = "/data/jamesmit/hf_cache"
PORT = 8001  # avoid clashing with any ATOM server on 8000
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}/v1/completions"
ISL = 1024
OSL = 8192

_DEFAULT_CONFIGS = "1:0,2:0,4:1,8:1"
TP_CONFIGS = []
for item in os.environ.get("TP_CONFIGS", _DEFAULT_CONFIGS).split(","):
    tp_str, ep_str = item.split(":")
    TP_CONFIGS.append((int(tp_str), bool(int(ep_str))))
CONCURRENCIES = [1, 8]

REPO = Path("/home/AMD/jamesmit/apps/ATOM")
CTNAME = "vllm_bench_step3p5"


def docker_run_cmd(tp: int, ep: bool) -> list[str]:
    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", CTNAME,
        "--network=host",
        "--ipc=host",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", "16g",
        # Pin to first `tp` GPUs
        "-e", f"HIP_VISIBLE_DEVICES={','.join(str(i) for i in range(tp))}",
        "-e", "HF_HUB_OFFLINE=1",
        # Step3.5 uses swigluoai activation; AITER FP8 MoE backend in vLLM 0.16
        # does not support it. Disable AITER for MoE so vLLM falls back to
        # the Triton FP8 MoE kernels. AITER is still used for attention.
        "-e", "VLLM_ROCM_USE_AITER_MOE=0",
        "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
        VLLM_IMAGE,
        "vllm", "serve", MODEL_CT,
        "--served-model-name", MODEL_CT,
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--tensor-parallel-size", str(tp),
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.92",
        "--max-model-len", "16384",  # plenty for ISL+OSL=9216
        # FULL CUDA-graph capture caused a GPU memory access fault on this
        # model (Step3p5 + AITER attention + fp8 KV). For TP=1 PIECEWISE works.
        # For TP>=2 even PIECEWISE faults during the first decode step in this
        # vLLM 0.16.1.dev / ROCm 7.12 build, so fall back to eager. Note this
        # makes the multi-GPU vLLM numbers a less direct comparison to ATOM
        # level=3 (CUDA graphs on); the TP=1 number IS apples-to-apples.
        *(["--compilation-config", '{"cudagraph_mode":"PIECEWISE"}']
          if tp == 1 else ["--enforce-eager"]),
        # Multi-threaded shard loader: ~3x faster weight load on local NVMe.
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 16}',
        "--disable-log-requests",
        # vLLM v0.16 custom_all_reduce_hip kernel hits 'invalid device pointer'
        # on this ROCm 7.12 build for TP>=2; fall back to NCCL/RCCL all-reduce.
        "--disable-custom-all-reduce",
    ]
    if ep:
        cmd.append("--enable-expert-parallel")
    return cmd


def kill_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", CTNAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_server(tp: int, ep: bool) -> str:
    """Start the vLLM container in detached mode; return container id."""
    kill_container()
    suffix = f"tp{tp}" + ("_ep" if ep else "")
    log_path = REPO / f"vllm_bench_{suffix}.log"
    cmd = docker_run_cmd(tp, ep)
    print(f"[bench] launching vLLM TP={tp} EP={ep}, log -> {log_path}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"docker run failed: {res.returncode}")
    cid = res.stdout.strip()
    # Tee container logs to a file for postmortem
    log_f = open(log_path, "wb")
    subprocess.Popen(
        ["docker", "logs", "-f", CTNAME],
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    return cid


def container_alive() -> bool:
    res = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CTNAME],
        capture_output=True, text=True,
    )
    return res.returncode == 0 and res.stdout.strip() == "true"


def wait_ready(timeout: float = 1500.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not container_alive():
            return False
        try:
            r = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if r.status_code == 200:
                m = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=2)
                if m.status_code == 200 and MODEL_CT in m.text:
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def make_prompt(n_tokens: int) -> str:
    return " ".join(["the"] * n_tokens)


def one_request_streaming(prompt: str, max_tokens: int) -> dict:
    """Stream a single completion, capture TTFT, derive TPOT from wall - ttft."""
    payload = {
        "model": MODEL_CT,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft_s = None
    completion_tokens = 0
    prompt_tokens = 0
    with requests.post(URL, json=payload, timeout=1200, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            # First chunk with text -> TTFT
            choices = obj.get("choices") or []
            if ttft_s is None and choices and choices[0].get("text"):
                ttft_s = time.time() - t0
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    wall_s = time.time() - t0
    if ttft_s is None:
        ttft_s = wall_s
    if completion_tokens > 1:
        tpot_s = (wall_s - ttft_s) / (completion_tokens - 1)
    else:
        tpot_s = 0.0
    return {
        "ttft_ms": ttft_s * 1000.0,
        "tpot_ms": tpot_s * 1000.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
    }


def run_concurrent(prompt: str, max_tokens: int, concurrency: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request_streaming, prompt, max_tokens) for _ in range(concurrency)]
        return [f.result() for f in as_completed(futs)]


def summarize(results: list[dict], wall_s: float) -> dict:
    n = len(results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / n
    avg_tpot = sum(r["tpot_ms"] for r in results) / n
    total_in = sum(r["prompt_tokens"] for r in results)
    total_out = sum(r["completion_tokens"] for r in results)
    decode_toks = (1000.0 / avg_tpot) if avg_tpot > 0 else 0.0
    output_toks = total_out / wall_s if wall_s > 0 else 0.0
    total_toks = (total_in + total_out) / wall_s if wall_s > 0 else 0.0
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
    rows: list[dict] = []
    launch_server(tp, ep)
    try:
        ok = wait_ready()
        if not ok:
            print(f"[bench] {label} server failed to come up", flush=True)
            return rows
        prompt = make_prompt(ISL)
        print(f"[bench] {label} warming up (concurrency=1, max_tokens=64)...", flush=True)
        run_concurrent(prompt, 64, 1)
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
            row = {
                "engine": "vllm",
                "tp": tp, "ep": ep, "isl": ISL, "osl": OSL,
                "concurrency": c, **summary,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        kill_container()
        time.sleep(5)
    return rows


def main() -> int:
    all_rows: list[dict] = []
    for tp, ep in TP_CONFIGS:
        all_rows.extend(bench_tp(tp, ep))

    out_path = REPO / "bench_step3p5_tp_results_vllm.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n[bench] wrote {out_path}", flush=True)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
