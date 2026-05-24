#!/usr/bin/env python3
"""Compare final-position logits for Step-3.5-Flash across ATOM, vLLM, and SGLang.

Usage:
  # Run inside each engine's environment to dump per-engine JSON:
  python tools/compare_step3p5_logits.py --internal-engine atom --max-prompts 1
  python tools/compare_step3p5_logits.py --internal-engine vllm --max-prompts 1
  python tools/compare_step3p5_logits.py --internal-engine sglang --max-prompts 1

  # Orchestrate all engines and write comparison JSON:
  python tools/compare_step3p5_logits.py atom vllm --max-prompts 2
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from step3p5_compare_prompts import PROMPTS

os.environ.setdefault("DOCKER_API_VERSION", "1.43")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
MODEL_ATOM = MODEL_HOST
MODEL_CT = "/models/Step-3.5-Flash-FP8"
REPO = Path("/home/AMD/jamesmit/apps/ATOM")
VENV = REPO / ".venv-gfx942"
VENV_CORE = VENV / "lib/python3.12/site-packages/_rocm_sdk_core"
VENV_DEVEL = VENV / "lib/python3.12/site-packages/_rocm_sdk_devel"
VENV_LIBS = VENV / "lib/python3.12/site-packages/_rocm_sdk_libraries_gfx94X_dcgpu/lib"
VLLM_IMAGE = (
    "rocm/vllm:rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"
)
SGL_IMAGE = "rocm/sgl-dev:v0.5.10.post1-rocm720-mi30x-20260503"
VLLM_CTNAME = "vllm_compare_step3p5_logits"
SGLANG_CTNAME = "sglang_compare_step3p5_logits"
TOP_K = 10

ATOM_PORT = 8000

# ---------------------------------------------------------------------------
# Environment / process helpers
# ---------------------------------------------------------------------------
def atom_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{VENV / 'bin'}:{VENV_CORE / 'bin'}:{VENV_DEVEL / 'bin'}:{env.get('PATH', '')}"
    env["ROCM_HOME"] = str(VENV_DEVEL)
    env["ROCM_PATH"] = str(VENV_DEVEL)
    env["HIP_PATH"] = str(VENV_CORE)
    env["HIP_DEVICE_LIB_PATH"] = str(VENV_CORE / "lib/llvm/amdgcn/bitcode")
    env["LD_LIBRARY_PATH"] = (
        f"{VENV_LIBS}:{VENV_CORE / 'lib'}:{VENV_DEVEL / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["LIBRARY_PATH"] = (
        f"{VENV_CORE / 'lib'}:{VENV_DEVEL / 'lib'}:{VENV_LIBS}:{env.get('LIBRARY_PATH', '')}"
    )
    env["CPLUS_INCLUDE_PATH"] = (
        f"{VENV_DEVEL / 'include'}:{VENV_CORE / 'include'}:{env.get('CPLUS_INCLUDE_PATH', '')}"
    )
    env["HF_HOME"] = "/data/jamesmit/hf_cache"
    env["HF_HUB_CACHE"] = "/data/jamesmit/hf_cache"
    env["TRANSFORMERS_CACHE"] = "/data/jamesmit/hf_cache"
    env["TRITON_CACHE_DIR"] = "/data/jamesmit/triton_cache"
    env["AITER_LOG_LEVEL"] = "WARNING"
    env["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    return env


def host_cpu_count() -> int:
    try:
        with open("/proc/cpuinfo") as f:
            return sum(1 for line in f if line.startswith("processor"))
    except OSError:
        return os.cpu_count() or 1


def kill_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


# ---------------------------------------------------------------------------
# Summary / comparison helpers
# ---------------------------------------------------------------------------
def extract_topk(logits: list[float], top_k: int) -> list[dict]:
    indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:top_k]
    return [{"token_id": i, "logit": float(logits[i])} for i in indices]


def vector_stats(logits: list[float]) -> dict:
    finite = [v for v in logits if math.isfinite(v)]
    if not finite:
        return {"argmax_token_id": 0, "max_logit": float("-inf"),
                "min_logit": float("-inf"), "mean_logit": float("nan"),
                "l2_norm": 0.0, "finite_count": 0, "total_count": len(logits)}
    argmax_id = max(range(len(logits)), key=lambda i: logits[i])
    return {
        "argmax_token_id": int(argmax_id),
        "max_logit": float(max(finite)),
        "min_logit": float(min(finite)),
        "mean_logit": float(sum(finite) / len(finite)),
        "l2_norm": float(math.sqrt(sum(v * v for v in finite))),
        "finite_count": len(finite),
        "total_count": len(logits),
    }


def summarize_prompt(prompt_id: str, prompt: str, input_ids: list[int],
                     logits: list[float], is_sparse: bool = False) -> dict:
    return {
        "id": prompt_id,
        "prompt": prompt,
        "prompt_token_ids": input_ids,
        "prompt_length": len(input_ids),
        "is_sparse": is_sparse,
        "logit_summary": vector_stats(logits),
        "top_tokens": extract_topk(logits, TOP_K),
        "final_position_logits": logits,
    }


def compare_topk(a_topk: list[dict], b_topk: list[dict]) -> dict:
    a_ids = {t["token_id"] for t in a_topk}
    b_ids = {t["token_id"] for t in b_topk}
    overlap = a_ids & b_ids
    a_argmax = a_topk[0]["token_id"] if a_topk else None
    b_argmax = b_topk[0]["token_id"] if b_topk else None
    return {
        "argmax_match": a_argmax == b_argmax,
        "topk_overlap_count": len(overlap),
        "topk_overlap_ids": sorted(overlap),
        "a_argmax": a_argmax,
        "b_argmax": b_argmax,
    }


def compare_logits_vectors(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b):
        return {"error": f"length mismatch: {len(a)} vs {len(b)}"}
    pairs = [(av, bv) for av, bv in zip(a, b) if math.isfinite(av) and math.isfinite(bv)]
    if not pairs:
        return {"error": "no finite pairs to compare"}
    diffs = [abs(av - bv) for av, bv in pairs]
    dot = sum(av * bv for av, bv in pairs)
    norm_a = math.sqrt(sum(av * av for av, _ in pairs))
    norm_b = math.sqrt(sum(bv * bv for _, bv in pairs))
    cosine = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else float("nan")
    return {
        "finite_pairs": len(pairs),
        "max_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(sum(diffs) / len(diffs)),
        "cosine_similarity": float(cosine),
    }


# ===================================================================
# Internal engine modes — run inside respective environments
# ===================================================================

def run_internal_atom(prompts: list[dict], model_path: str) -> dict:
    """ATOM via its OpenAI-compatible server (greedy argmax only).

    ATOM does not expose logprobs through its server API, so we launch
    the server, send each prompt with temperature=0 max_tokens=1, and
    record the first generated token as the argmax.  The output is
    marked as argmax-only (is_sparse=True) with a synthetic logits
    vector containing only the argmax entry.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    port = ATOM_PORT
    log_path = REPO / "compare_step3p5_atom_logits.log"
    log_file = open(log_path, "wb")
    cmd = [
        str(VENV / "bin/python"), "-m", "atom.entrypoints.openai_server",
        "--model", model_path,
        "--kv_cache_dtype", "fp8",
        "-tp", "1",
        "--gpu-memory-utilization", "0.92",
        "--port", str(port),
    ]
    print("[atom] starting ATOM server...", file=sys.stderr, flush=True)
    proc = subprocess.Popen(
        cmd, env=atom_env(), cwd=str(REPO),
        stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        ready = _wait_server(f"http://127.0.0.1:{port}", model_path, timeout=600)
        if not ready:
            raise RuntimeError("ATOM server failed to become ready")

        vocab_size = tokenizer.vocab_size
        records = []
        for item in prompts:
            tokens = tokenizer.encode(item["prompt"], add_special_tokens=True)
            resp = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={
                    "model": model_path,
                    "prompt": item["prompt"],
                    "temperature": 0.0,
                    "max_tokens": 1,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            gen_text = data["choices"][0]["text"]
            gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            argmax_id = gen_token_ids[0] if gen_token_ids else 0

            logits = [float("-inf")] * vocab_size
            logits[argmax_id] = 0.0

            records.append(summarize_prompt(
                item["id"], item["prompt"], tokens, logits, is_sparse=True,
            ))
            print(f"[atom] '{item['id']}' argmax={argmax_id}", file=sys.stderr, flush=True)

    finally:
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

    return {
        "engine": "atom",
        "model": model_path,
        "prompt_count": len(records),
        "records": records,
    }


def _wait_server(base_url: str, model_name: str, timeout: float = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = requests.get(f"{base_url}/health", timeout=2)
            if h.status_code == 200:
                m = requests.get(f"{base_url}/v1/models", timeout=2)
                if m.status_code == 200 and model_name in m.text:
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def run_internal_vllm(prompts: list[dict], model_path: str) -> dict:
    """vLLM offline API with prompt_logprobs (sparse top-k only)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=False,
        gpu_memory_utilization=0.92,
        kv_cache_dtype="fp8",
        max_model_len=16384,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        compilation_config={"cudagraph_mode": "PIECEWISE"},
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=TOP_K,
        logprobs=TOP_K,
        detokenize=False,
    )
    outputs = llm.generate(
        [item["prompt"] for item in prompts], sampling_params, use_tqdm=False,
    )

    vocab_size = llm.llm_engine.get_tokenizer().vocab_size
    records = []
    for item, output in zip(prompts, outputs):
        prompt_token_ids = list(output.prompt_token_ids)
        prompt_logprobs = output.prompt_logprobs or []
        if not prompt_logprobs:
            raise RuntimeError("vLLM did not return prompt_logprobs")
        final_logprob_map = prompt_logprobs[-1] or {}
        if not final_logprob_map:
            raise RuntimeError("vLLM final prompt logprobs were empty")
        logits = [float("-inf")] * vocab_size
        for token_id, logprob_obj in final_logprob_map.items():
            logits[int(token_id)] = float(logprob_obj.logprob)
        records.append(summarize_prompt(
            item["id"], item["prompt"], prompt_token_ids, logits, is_sparse=True,
        ))

    return {
        "engine": "vllm",
        "model": model_path,
        "prompt_count": len(records),
        "records": records,
    }


def run_internal_sglang(prompts: list[dict], model_path: str) -> dict:
    """SGLang offline Engine with return_logprob for sparse top-k."""
    from sglang.srt.entrypoints.engine import Engine

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)
    vocab_size = model_config.get("vocab_size", 128896)

    llm = Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=0.88,
        kv_cache_dtype="auto",
        context_length=16384,
        disable_cuda_graph=True,
    )

    records = []
    for item in prompts:
        out = llm.generate(
            prompt=item["prompt"],
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=TOP_K,
        )
        meta = out.get("meta_info", {})
        input_ids = meta.get("prompt_tokens", 0)
        input_top = meta.get("input_top_logprobs", [])
        output_top = meta.get("output_top_logprobs", [])

        if input_top:
            final_map = input_top[-1]
        elif output_top:
            final_map = output_top[0]
        else:
            raise RuntimeError("SGLang returned no logprobs at all")

        logits = [float("-inf")] * vocab_size
        for entry in final_map:
            tok_id = entry[1]
            logprob_val = entry[0]
            logits[int(tok_id)] = float(logprob_val)

        prompt_token_count = meta.get("prompt_tokens", 0)
        records.append(summarize_prompt(
            item["id"], item["prompt"], list(range(prompt_token_count)),
            logits, is_sparse=True,
        ))

    llm.shutdown()

    return {
        "engine": "sglang",
        "model": model_path,
        "prompt_count": len(records),
        "records": records,
    }


# ===================================================================
# Orchestrator — launch engines via subprocess/docker, collect JSON
# ===================================================================
def _extract_json_from_output(stdout: str) -> dict:
    """Extract JSON payload delimited by sentinel markers from noisy stdout."""
    begin = "__LOGITS_JSON_BEGIN__"
    end = "__LOGITS_JSON_END__"
    b = stdout.find(begin)
    e = stdout.find(end)
    if b == -1 or e == -1:
        # Fallback: try parsing last non-empty line as JSON
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError(f"No JSON found in stdout (len={len(stdout)})")
    payload = stdout[b + len(begin):e].strip()
    return json.loads(payload)


def run_engine_subprocess(engine: str, max_prompts: int | None) -> dict:
    """Launch internal mode in appropriate env and capture JSON stdout."""
    extra_args = []
    if max_prompts:
        extra_args = ["--max-prompts", str(max_prompts)]

    if engine == "atom":
        cmd = [
            str(VENV / "bin/python"),
            str(REPO / "tools/compare_step3p5_logits.py"),
            "--internal-engine", "atom",
        ] + extra_args
        print(f"[orchestrator] launching ATOM subprocess...", flush=True)
        result = subprocess.run(
            cmd, env=atom_env(), cwd=str(REPO),
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"[orchestrator] ATOM stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"ATOM subprocess failed (rc={result.returncode})")
        return _extract_json_from_output(result.stdout)

    elif engine == "vllm":
        kill_container(VLLM_CTNAME)
        cmd = [
            "docker", "run", "--rm",
            "--name", VLLM_CTNAME,
            "--network=host", "--ipc=host",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video",
            "--cap-add=SYS_PTRACE",
            "--security-opt", "seccomp=unconfined",
            "--shm-size", "16g",
            "-e", f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '0')}",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "VLLM_ROCM_USE_AITER_MOE=0",
            "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
            "-v", f"{REPO}:/workspace/ATOM",
            VLLM_IMAGE,
            "python3", "/workspace/ATOM/tools/compare_step3p5_logits.py",
            "--internal-engine", "vllm",
        ] + extra_args
        print(f"[orchestrator] launching vLLM container...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"[orchestrator] vLLM stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"vLLM container failed (rc={result.returncode})")
        kill_container(VLLM_CTNAME)
        return _extract_json_from_output(result.stdout)

    elif engine == "sglang":
        kill_container(SGLANG_CTNAME)
        host_cpus = host_cpu_count()
        cmd = [
            "docker", "run", "--rm",
            "--name", SGLANG_CTNAME,
            "--network=host", "--ipc=host",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video",
            "--cap-add=SYS_PTRACE",
            "--security-opt", "seccomp=unconfined",
            "--shm-size", "32g",
            "--cpuset-cpus", f"0-{host_cpus - 1}",
            "-e", f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '0')}",
            "-e", "SGLANG_USE_AITER=1",
            "-e", "SGLANG_SET_CPU_AFFINITY=0",
            "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
            "-v", f"{REPO}:/workspace/ATOM",
            "-v", f"{REPO}/tools/sglang_patches/step3p5.py:/sgl-workspace/sglang/python/sglang/srt/models/step3p5.py:ro",
            "-v", f"{REPO}/tools/sglang_patches/fp8.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py:ro",
            SGL_IMAGE,
            "python3", "/workspace/ATOM/tools/compare_step3p5_logits.py",
            "--internal-engine", "sglang",
        ] + extra_args
        print(f"[orchestrator] launching SGLang container...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"[orchestrator] SGLang stderr:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"SGLang container failed (rc={result.returncode})")
        kill_container(SGLANG_CTNAME)
        return _extract_json_from_output(result.stdout)

    else:
        raise ValueError(f"Unknown engine: {engine}")


def build_comparison(engine_results: dict[str, dict]) -> dict:
    """Build pairwise comparison between engines."""
    engines = sorted(engine_results.keys())
    comparisons = {}
    for i, eng_a in enumerate(engines):
        for eng_b in engines[i + 1:]:
            pair_key = f"{eng_a}_vs_{eng_b}"
            a_records = engine_results[eng_a]["records"]
            b_records = engine_results[eng_b]["records"]
            pair_results = []
            for a_rec, b_rec in zip(a_records, b_records):
                prompt_id = a_rec["id"]
                topk_cmp = compare_topk(a_rec["top_tokens"], b_rec["top_tokens"])
                both_sparse = a_rec["is_sparse"] and b_rec["is_sparse"]
                logit_cmp = {}
                if not both_sparse:
                    logit_cmp = compare_logits_vectors(
                        a_rec["final_position_logits"],
                        b_rec["final_position_logits"],
                    )
                pair_results.append({
                    "prompt_id": prompt_id,
                    "topk_comparison": topk_cmp,
                    "logit_comparison": logit_cmp,
                })
            comparisons[pair_key] = pair_results
    return comparisons


# ===================================================================
# CLI
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare Step-3.5-Flash logits across engines")
    p.add_argument("engines", nargs="*", default=["atom", "vllm"],
                    help="Engines to compare in orchestrator mode")
    p.add_argument("--max-prompts", type=int, default=None,
                    help="Limit number of prompts")
    p.add_argument("--internal-engine", choices=["atom", "vllm", "sglang"],
                    default=None, help="Run in internal single-engine mode")
    p.add_argument("--model", default=None,
                    help="Model path override (auto-detected per environment)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    prompts = PROMPTS[:args.max_prompts] if args.max_prompts else PROMPTS

    if args.internal_engine:
        model = args.model
        if model is None:
            if args.internal_engine == "atom":
                model = MODEL_ATOM
            else:
                model = MODEL_CT

        if args.internal_engine == "atom":
            result = run_internal_atom(prompts, model)
        elif args.internal_engine == "vllm":
            result = run_internal_vllm(prompts, model)
        elif args.internal_engine == "sglang":
            result = run_internal_sglang(prompts, model)
        else:
            raise ValueError(args.internal_engine)

        print("__LOGITS_JSON_BEGIN__")
        print(json.dumps(result))
        print("__LOGITS_JSON_END__")
        return 0

    # Orchestrator mode
    engine_results = {}
    for engine in args.engines:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[orchestrator] running {engine}...", flush=True)
        print(f"{'=' * 60}", flush=True)
        engine_results[engine] = run_engine_subprocess(engine, args.max_prompts)
        print(f"[orchestrator] {engine} done — {engine_results[engine]['prompt_count']} prompts", flush=True)

    comparisons = build_comparison(engine_results)

    output = {
        "top_k": TOP_K,
        "prompt_count": len(prompts),
        "engines": list(engine_results.keys()),
        "comparisons": comparisons,
        "per_engine": {
            eng: {
                "model": data["model"],
                "records": [
                    {
                        "id": r["id"],
                        "prompt_length": r["prompt_length"],
                        "is_sparse": r["is_sparse"],
                        "logit_summary": r["logit_summary"],
                        "top_tokens": r["top_tokens"],
                    }
                    for r in data["records"]
                ],
            }
            for eng, data in engine_results.items()
        },
    }

    out_path = REPO / "step3p5_logits_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[orchestrator] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
