import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from step3p5_compare_prompts import PROMPTS

os.environ.setdefault("DOCKER_API_VERSION", "1.43")

MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
MODEL_ATOM = MODEL_HOST
MODEL_CT = "/models/Step-3.5-Flash-FP8"
ATOM_PORT = 8000
VLLM_PORT = 8001
SGLANG_PORT = 8002
HOST = "127.0.0.1"
REPO = Path("/home/AMD/jamesmit/apps/ATOM")
VENV = REPO / ".venv-gfx942"
VENV_CORE = VENV / "lib/python3.12/site-packages/_rocm_sdk_core"
VENV_DEVEL = VENV / "lib/python3.12/site-packages/_rocm_sdk_devel"
VENV_LIBS = VENV / "lib/python3.12/site-packages/_rocm_sdk_libraries_gfx94X_dcgpu/lib"
VLLM_IMAGE = (
    "rocm/vllm:rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"
)
SGL_IMAGE = "rocm/sgl-dev:v0.5.10.post1-rocm720-mi30x-20260503"
VLLM_CTNAME = "vllm_compare_step3p5"
SGLANG_CTNAME = "sglang_compare_step3p5"
MAX_TOKENS = 256
TEMPERATURE = 0.0
def atom_env() -> dict:
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
    env["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    return env


def host_cpu_count() -> int:
    try:
        with open("/proc/cpuinfo") as file:
            return sum(1 for line in file if line.startswith("processor"))
    except OSError:
        return os.cpu_count() or 1


def kill_atom(proc: subprocess.Popen) -> None:
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


def kill_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def wait_http_ready(port: int, model_name: str, timeout: float = 1800.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            health = requests.get(f"http://{HOST}:{port}/health", timeout=2)
            if health.status_code == 200:
                models = requests.get(f"http://{HOST}:{port}/v1/models", timeout=2)
                if models.status_code == 200 and model_name in models.text:
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def container_alive(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def wait_container_ready(name: str, port: int, model_name: str, timeout: float = 1800.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not container_alive(name):
            return False
        try:
            health = requests.get(f"http://{HOST}:{port}/health", timeout=2)
            if health.status_code == 200:
                models = requests.get(f"http://{HOST}:{port}/v1/models", timeout=2)
                if models.status_code == 200 and model_name in models.text:
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def request_completion(port: int, model_name: str, prompt: str) -> dict:
    response = requests.post(
        f"http://{HOST}:{port}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        timeout=1200,
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    return {
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
    }


def launch_atom() -> subprocess.Popen:
    log_path = REPO / "compare_step3p5_atom.log"
    log_file = open(log_path, "wb")
    cmd = [
        str(VENV / "bin/python"),
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        MODEL_ATOM,
        "--kv_cache_dtype",
        "fp8",
        "-tp",
        "1",
        "--gpu-memory-utilization",
        "0.92",
        "--port",
        str(ATOM_PORT),
    ]
    return subprocess.Popen(
        cmd,
        env=atom_env(),
        cwd=str(REPO),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def launch_vllm() -> None:
    kill_container(VLLM_CTNAME)
    log_path = REPO / "compare_step3p5_vllm.log"
    log_file = open(log_path, "wb")
    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", VLLM_CTNAME,
        "--network=host",
        "--ipc=host",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", "16g",
        "-e", "HIP_VISIBLE_DEVICES=0",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "VLLM_ROCM_USE_AITER_MOE=0",
        "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
        VLLM_IMAGE,
        "vllm", "serve", MODEL_CT,
        "--served-model-name", MODEL_CT,
        "--port", str(VLLM_PORT),
        "--host", "0.0.0.0",
        "--tensor-parallel-size", "1",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", "0.92",
        "--max-model-len", "16384",
        "--compilation-config", '{"cudagraph_mode":"PIECEWISE"}',
        "--model-loader-extra-config", '{"enable_multithread_load": true, "num_threads": 16}',
        "--disable-log-requests",
        "--disable-custom-all-reduce",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    subprocess.Popen(
        ["docker", "logs", "-f", VLLM_CTNAME],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def launch_sglang() -> None:
    kill_container(SGLANG_CTNAME)
    log_path = REPO / "compare_step3p5_sglang.log"
    log_file = open(log_path, "wb")
    host_cpus = host_cpu_count()
    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", SGLANG_CTNAME,
        "--network=host",
        "--ipc=host",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", "32g",
        "--cpuset-cpus", f"0-{host_cpus - 1}",
        "-e", "HIP_VISIBLE_DEVICES=0",
        "-e", "SGLANG_USE_AITER=1",
        "-e", "SGLANG_SET_CPU_AFFINITY=0",
        "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
        "-v", f"{REPO}/tools/sglang_patches/step3p5.py:/sgl-workspace/sglang/python/sglang/srt/models/step3p5.py:ro",
        "-v", f"{REPO}/tools/sglang_patches/fp8.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py:ro",
        SGL_IMAGE,
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_CT,
        "--served-model-name", MODEL_CT,
        "--host", "0.0.0.0",
        "--port", str(SGLANG_PORT),
        "--tp", "1",
        "--mem-fraction-static", "0.92",
        "--context-length", "16384",
        "--trust-remote-code",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    subprocess.Popen(
        ["docker", "logs", "-f", SGLANG_CTNAME],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def run_engine(engine: str) -> list[dict]:
    records = []
    if engine == "atom":
        proc = launch_atom()
        try:
            if not wait_http_ready(ATOM_PORT, MODEL_ATOM):
                raise RuntimeError("ATOM server failed to become ready")
            for item in PROMPTS:
                result = request_completion(ATOM_PORT, MODEL_ATOM, item["prompt"])
                records.append({"id": item["id"], "prompt": item["prompt"], **result})
        finally:
            kill_atom(proc)
    elif engine == "vllm":
        launch_vllm()
        try:
            if not wait_container_ready(VLLM_CTNAME, VLLM_PORT, MODEL_CT):
                raise RuntimeError("vLLM server failed to become ready")
            for item in PROMPTS:
                result = request_completion(VLLM_PORT, MODEL_CT, item["prompt"])
                records.append({"id": item["id"], "prompt": item["prompt"], **result})
        finally:
            kill_container(VLLM_CTNAME)
    elif engine == "sglang":
        launch_sglang()
        try:
            if not wait_container_ready(SGLANG_CTNAME, SGLANG_PORT, MODEL_CT):
                raise RuntimeError("SGLang server failed to become ready")
            for item in PROMPTS:
                result = request_completion(SGLANG_PORT, MODEL_CT, item["prompt"])
                records.append({"id": item["id"], "prompt": item["prompt"], **result})
        finally:
            kill_container(SGLANG_CTNAME)
    else:
        raise ValueError(engine)
    return records


def compare_outputs(engine_results: dict[str, list[dict]]) -> dict:
    prompt_ids = [item["id"] for item in PROMPTS]
    comparisons = []
    engines = list(engine_results.keys())
    exact_match_count = 0
    for prompt_id in prompt_ids:
        by_engine = {}
        for engine in engines:
            record = next(item for item in engine_results[engine] if item["id"] == prompt_id)
            by_engine[engine] = record
        texts = {engine: normalize_text(record["text"]) for engine, record in by_engine.items()}
        unique_texts = list(dict.fromkeys(texts.values()))
        exact_match = len(unique_texts) == 1
        if exact_match:
            exact_match_count += 1
        comparisons.append(
            {
                "id": prompt_id,
                "exact_match": exact_match,
                "outputs": {
                    engine: {
                        "text": by_engine[engine]["text"],
                        "finish_reason": by_engine[engine]["finish_reason"],
                        "usage": by_engine[engine]["usage"],
                    }
                    for engine in engines
                },
            }
        )
    return {
        "engines": engines,
        "prompt_count": len(prompt_ids),
        "exact_match_count": exact_match_count,
        "comparisons": comparisons,
    }


def main() -> int:
    requested_engines = sys.argv[1:] or ["atom", "vllm"]
    engine_results = {}
    for engine in requested_engines:
        print(f"[compare] running {engine}", flush=True)
        engine_results[engine] = run_engine(engine)
    summary = compare_outputs(engine_results)
    output = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "prompts": PROMPTS,
        "results": engine_results,
        "summary": summary,
    }
    out_path = REPO / "step3p5_output_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[compare] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
