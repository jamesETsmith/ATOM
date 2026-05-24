"""Benchmark Step-3.5-Flash-FP8 on SGLang (rocm/sgl-dev image) at TP=1,2,4,8.

Mirrors tools/bench_step3p5_tp_vllm.py but launches SGLang inside a docker
container. Same model, ISL, OSL, and concurrencies for direct comparison.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SGL_IMAGE = "rocm/sgl-dev:v0.5.10.post1-rocm720-mi30x-20260503"
MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
MODEL_CT = "/models/Step-3.5-Flash-FP8"
PORT = 8002  # avoid clashing with anything else
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
CTNAME = "sglang_bench_step3p5"


def host_cpu_count() -> int:
    # /proc/cpuinfo reports the true host CPU count regardless of cgroup
    # cpuset restrictions imposed on this shell.
    try:
        with open("/proc/cpuinfo") as f:
            return sum(1 for line in f if line.startswith("processor"))
    except OSError:
        return os.cpu_count() or 1


HOST_CPUS = host_cpu_count()


def docker_run_cmd(tp: int, ep: bool) -> list[str]:
    cmd = [
        "docker", "run", "-d",
        "--name", CTNAME,
        "--network=host",
        "--ipc=host",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", "32g",
        # Give container access to all host CPUs. SGLang's set_gpu_proc_affinity
        # picks CPU IDs > 8 and crashes if the cgroup cpuset is restricted.
        "--cpuset-cpus", f"0-{HOST_CPUS - 1}",
        "-e", f"HIP_VISIBLE_DEVICES={','.join(str(i) for i in range(tp))}",
        # Master AITER switch: enables AITER MoE/GEMM kernels.
        "-e", "SGLANG_USE_AITER=1",
        # The rocm/sgl-dev image bakes SGLANG_SET_CPU_AFFINITY=1, but the
        # docker daemon inherits this shell's cgroup cpuset (only CPUs 1-8),
        # so SGLang's per-rank affinity setter crashes for rank > 0 when it
        # tries to bind to high CPU IDs. Disable affinity binding entirely.
        "-e", "SGLANG_SET_CPU_AFFINITY=0",
        "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
        # Patched step3p5.py: filters scale-only placeholders from the final
        # "unloaded params" assert so mixed-precision FP8+bf16 checkpoints
        # (qkv_proj, share_expert, dense MLP all bf16) load successfully.
        "-v", f"{REPO}/tools/sglang_patches/step3p5.py:/sgl-workspace/sglang/python/sglang/srt/models/step3p5.py:ro",
        # Patched fp8.py: expand merged fused-linear names (gate_up_proj,
        # qkv_proj) in modules_to_not_convert into their unfused shard
        # names so is_layer_skipped() correctly skips bf16 layers from
        # FP8 quantization. Without this, share_expert.gate_up_proj at
        # TP=4 (320) and TP=8 (160) trips block-quant validation.
        "-v", f"{REPO}/tools/sglang_patches/fp8.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py:ro",
        SGL_IMAGE,
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_CT,
        "--served-model-name", MODEL_CT,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--tp", str(tp),
        # Do NOT pass --kv-cache-dtype or --quantization. Step-3.5-Flash-FP8 is a
        # mixed-precision checkpoint (FP8 routed experts, bf16 everything else).
        # The HF model card relies on autodetection; passing fp8 flags trips the
        # bespoke step3p5.py loader's `weight_scale_inv` assertion on bf16 modules.
        "--mem-fraction-static", "0.92",
        "--context-length", "16384",  # plenty for ISL+OSL=9216
        "--trust-remote-code",
    ]
    if ep:
        # Set EP size equal to TP for full expert parallelism.
        cmd.extend(["--ep", str(tp)])
    return cmd


def kill_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", CTNAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_server(tp: int, ep: bool) -> str:
    kill_container()
    suffix = f"tp{tp}" + ("_ep" if ep else "")
    log_path = REPO / f"sglang_bench_{suffix}.log"
    cmd = docker_run_cmd(tp, ep)
    print(f"[bench] launching SGLang TP={tp} EP={ep}, log -> {log_path}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"docker run failed: {res.returncode}")
    cid = res.stdout.strip()
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
            # SGLang uses /health for liveness and /v1/models for readiness.
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
    chunk_text_count = 0  # fallback when usage chunk is missing
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
            choices = obj.get("choices") or []
            # SGLang emits a non-null `text` (possibly empty string) on
            # every token chunk. Treat the first chunk that *contains* the
            # `text` key as the first-token marker, even if the string is
            # empty (some tokenizers split tokens that yield no visible
            # text fragment, e.g. byte-pair leading bytes).
            if choices and "text" in choices[0]:
                if ttft_s is None:
                    ttft_s = time.time() - t0
                chunk_text_count += 1
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    wall_s = time.time() - t0
    if ttft_s is None:
        ttft_s = wall_s
    if completion_tokens == 0:
        # Fallback: SGLang sometimes withholds usage chunk; use chunk count.
        completion_tokens = chunk_text_count
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
                "engine": "sglang",
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

    out_path = REPO / "bench_step3p5_tp_results_sglang.json"
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
