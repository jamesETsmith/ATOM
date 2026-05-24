#!/usr/bin/env python3
"""Comprehensive Step-3.5-Flash MTP & EP GPU validation suite.

Runs a series of tests against a live ATOM server and produces a
structured report.  Designed to be run on an 8x MI325X node with
the Step-3.5-Flash-FP8 checkpoint available.

Usage:
    .venv-gfx942/bin/python tools/gpu_validation_step3p5.py \
        --model /data/jamesmit/models/Step-3.5-Flash-FP8
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

PYTHON = sys.executable
PORT_BASE = 8200

# ── prompts ──────────────────────────────────────────────────────────

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a Python function to compute the nth Fibonacci number.",
    "The meaning of life is",
    "List 5 benefits of regular exercise:",
    "Translate 'Hello, how are you?' into Spanish.",
    "What is 15 * 23?",
    "Name the planets in our solar system.",
]

BENCH_CMD_TEMPLATE = [
    PYTHON, "-m", "atom.benchmarks.benchmark_serving",
    "--backend=vllm",
    "--dataset-name=random",
    "--random-range-ratio=0.8",
    "--request-rate=inf",
    "--ignore-eos",
    "--save-result",
    "--percentile-metrics=ttft,tpot,itl,e2el",
]

# ── helpers ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    status: str  # PASS / FAIL / SKIP / ERROR
    duration_s: float = 0.0
    detail: str = ""
    data: dict = field(default_factory=dict)


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def send_completion(base_url: str, prompt: str, max_tokens: int = 128,
                    model: str = "test") -> dict:
    import urllib.request
    payload = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def start_server(model: str, tp: int, port: int, extra_args: list[str],
                 log_path: str) -> subprocess.Popen:
    cmd = [
        PYTHON, "-m", "atom.entrypoints.openai_server",
        "--model", model, "-tp", str(tp),
        "--kv_cache_dtype", "fp8", "--port", str(port),
        "--trust-remote-code",
    ] + extra_args
    env = os.environ.copy()
    env["AITER_LOG_LEVEL"] = "WARNING"
    env["HF_HOME"] = "/data/jamesmit/hf_cache"
    env["TRANSFORMERS_CACHE"] = "/data/jamesmit/hf_cache"
    env["TRITON_CACHE_DIR"] = "/data/jamesmit/triton_cache"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc


def kill_server(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def collect_completions(base_url: str, prompts: list[str],
                        max_tokens: int = 128) -> list[dict]:
    results = []
    for p in prompts:
        try:
            resp = send_completion(base_url, p, max_tokens)
            text = resp["choices"][0]["text"].strip()
            results.append({"prompt": p, "output": text,
                           "tokens": resp.get("usage", {}).get("completion_tokens", -1)})
        except Exception as e:
            results.append({"prompt": p, "error": str(e)})
    return results


def run_benchmark(model: str, base_url: str, isl: int, osl: int,
                  conc: int, n_prompts: int, result_path: str) -> dict | None:
    cmd = BENCH_CMD_TEMPLATE + [
        f"--model={model}",
        f"--base-url={base_url}",
        f"--random-input-len={isl}",
        f"--random-output-len={osl}",
        f"--max-concurrency={conc}",
        f"--num-prompts={n_prompts}",
        f"--num-warmups={min(conc * 2, n_prompts // 2)}",
        f"--result-dir={Path(result_path).parent}",
        f"--result-filename={Path(result_path).name}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if Path(result_path).exists():
            return json.loads(Path(result_path).read_text())
    except Exception:
        pass
    return None


# ── test functions ───────────────────────────────────────────────────

def test_baseline_startup(model: str, log_dir: str) -> TestResult:
    """Test 1: Baseline TP=8 server startup (no MTP, no EP)."""
    port = PORT_BASE
    log = f"{log_dir}/baseline_tp8.log"
    t0 = time.time()
    proc = start_server(model, 8, port, [], log)
    try:
        ok = wait_for_server(f"http://localhost:{port}", timeout=600)
        dur = time.time() - t0
        if not ok:
            tail = Path(log).read_text()[-2000:]
            return TestResult("baseline_tp8_startup", "FAIL", dur,
                              f"Server did not start\n{tail}")
        results = collect_completions(f"http://localhost:{port}", TEST_PROMPTS[:3])
        errors = [r for r in results if "error" in r]
        return TestResult("baseline_tp8_startup", "PASS" if not errors else "FAIL",
                          dur, f"{len(results)-len(errors)}/{len(results)} prompts OK",
                          {"completions": results})
    finally:
        kill_server(proc)


def test_mtp_startup(model: str, log_dir: str) -> TestResult:
    """Test 2: MTP TP=8 server startup."""
    port = PORT_BASE + 1
    log = f"{log_dir}/mtp_tp8.log"
    t0 = time.time()
    proc = start_server(model, 8, port,
                        ["--method", "mtp", "--num-speculative-tokens", "1"], log)
    try:
        ok = wait_for_server(f"http://localhost:{port}", timeout=600)
        dur = time.time() - t0
        if not ok:
            tail = Path(log).read_text()[-2000:]
            return TestResult("mtp_tp8_startup", "FAIL", dur,
                              f"MTP server did not start\n{tail}")
        results = collect_completions(f"http://localhost:{port}", TEST_PROMPTS)
        errors = [r for r in results if "error" in r]
        return TestResult("mtp_tp8_startup", "PASS" if not errors else "FAIL",
                          dur, f"{len(results)-len(errors)}/{len(results)} prompts OK",
                          {"completions": results})
    finally:
        kill_server(proc)


def test_mtp_correctness(model: str, log_dir: str) -> TestResult:
    """Test 3: Compare MTP output vs baseline (first-word match)."""
    port_base = PORT_BASE + 2
    port_mtp = PORT_BASE + 3
    log_base = f"{log_dir}/correctness_baseline.log"
    log_mtp = f"{log_dir}/correctness_mtp.log"
    t0 = time.time()

    # Start baseline
    proc_base = start_server(model, 8, port_base, [], log_base)
    try:
        if not wait_for_server(f"http://localhost:{port_base}", 600):
            return TestResult("mtp_correctness", "FAIL", time.time()-t0,
                              "Baseline server failed to start")
        base_results = collect_completions(f"http://localhost:{port_base}",
                                           TEST_PROMPTS, 128)
    finally:
        kill_server(proc_base)

    # Start MTP
    proc_mtp = start_server(model, 8, port_mtp,
                            ["--method", "mtp", "--num-speculative-tokens", "1"],
                            log_mtp)
    try:
        if not wait_for_server(f"http://localhost:{port_mtp}", 600):
            return TestResult("mtp_correctness", "FAIL", time.time()-t0,
                              "MTP server failed to start")
        mtp_results = collect_completions(f"http://localhost:{port_mtp}",
                                          TEST_PROMPTS, 128)
    finally:
        kill_server(proc_mtp)

    dur = time.time() - t0

    # Compare first-word match
    matches = 0
    total = 0
    comparisons = []
    for b, m in zip(base_results, mtp_results):
        if "error" in b or "error" in m:
            comparisons.append({"prompt": b["prompt"], "match": "ERROR"})
            continue
        total += 1
        b_first = b["output"].split()[0] if b["output"] else ""
        m_first = m["output"].split()[0] if m["output"] else ""
        matched = b_first == m_first
        if matched:
            matches += 1
        comparisons.append({
            "prompt": b["prompt"],
            "base_first_word": b_first,
            "mtp_first_word": m_first,
            "match": matched,
        })

    ratio = f"{matches}/{total}"
    status = "PASS" if total > 0 and matches / total >= 0.75 else "FAIL"
    return TestResult("mtp_correctness", status, dur,
                      f"First-word match: {ratio}",
                      {"comparisons": comparisons, "match_ratio": ratio})


def test_ep_tp4_startup(model: str, log_dir: str) -> TestResult:
    """Test 4: EP with TP=4 startup (was crashing with IndexError)."""
    port = PORT_BASE + 4
    log = f"{log_dir}/ep_tp4.log"
    t0 = time.time()
    proc = start_server(model, 4, port, ["--enable-expert-parallel"], log)
    try:
        ok = wait_for_server(f"http://localhost:{port}", timeout=600)
        dur = time.time() - t0
        if not ok:
            tail = Path(log).read_text()[-2000:]
            return TestResult("ep_tp4_startup", "FAIL", dur,
                              f"EP TP=4 server failed\n{tail}")
        results = collect_completions(f"http://localhost:{port}", TEST_PROMPTS[:3])
        errors = [r for r in results if "error" in r]
        return TestResult("ep_tp4_startup", "PASS" if not errors else "FAIL",
                          dur, f"{len(results)-len(errors)}/{len(results)} prompts OK",
                          {"completions": results})
    finally:
        kill_server(proc)


def test_mtp_benchmark(model: str, log_dir: str) -> TestResult:
    """Test 5: MTP throughput benchmark (ISL=1024, OSL=1024, C=8)."""
    port = PORT_BASE + 5
    log = f"{log_dir}/bench_mtp.log"
    result_path = f"{log_dir}/bench_mtp_result.json"
    t0 = time.time()
    proc = start_server(model, 8, port,
                        ["--method", "mtp", "--num-speculative-tokens", "1"], log)
    try:
        if not wait_for_server(f"http://localhost:{port}", 600):
            return TestResult("mtp_benchmark", "FAIL", time.time()-t0,
                              "MTP server failed to start for benchmark")
        bench = run_benchmark(model, f"http://localhost:{port}",
                              1024, 1024, 8, 80, result_path)
        dur = time.time() - t0
        if bench is None:
            return TestResult("mtp_benchmark", "FAIL", dur,
                              "Benchmark did not produce results")
        return TestResult("mtp_benchmark", "PASS", dur,
                          f"output_throughput={bench.get('output_throughput', 'N/A')} tok/s",
                          {"benchmark": bench})
    finally:
        kill_server(proc)


def test_baseline_benchmark(model: str, log_dir: str) -> TestResult:
    """Test 6: Baseline throughput benchmark (ISL=1024, OSL=1024, C=8) for comparison."""
    port = PORT_BASE + 6
    log = f"{log_dir}/bench_baseline.log"
    result_path = f"{log_dir}/bench_baseline_result.json"
    t0 = time.time()
    proc = start_server(model, 8, port, [], log)
    try:
        if not wait_for_server(f"http://localhost:{port}", 600):
            return TestResult("baseline_benchmark", "FAIL", time.time()-t0,
                              "Baseline server failed to start for benchmark")
        bench = run_benchmark(model, f"http://localhost:{port}",
                              1024, 1024, 8, 80, result_path)
        dur = time.time() - t0
        if bench is None:
            return TestResult("baseline_benchmark", "FAIL", dur,
                              "Benchmark did not produce results")
        return TestResult("baseline_benchmark", "PASS", dur,
                          f"output_throughput={bench.get('output_throughput', 'N/A')} tok/s",
                          {"benchmark": bench})
    finally:
        kill_server(proc)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--log-dir", default="./validation_logs")
    parser.add_argument("--skip-bench", action="store_true",
                        help="Skip benchmark tests (faster)")
    args = parser.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    tests = [
        ("1_baseline_startup", test_baseline_startup),
        ("2_mtp_startup", test_mtp_startup),
        ("3_mtp_correctness", test_mtp_correctness),
        ("4_ep_tp4_startup", test_ep_tp4_startup),
    ]
    if not args.skip_bench:
        tests += [
            ("5_baseline_benchmark", test_baseline_benchmark),
            ("6_mtp_benchmark", test_mtp_benchmark),
        ]

    results: list[TestResult] = []
    for test_name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        try:
            result = test_fn(args.model, log_dir)
        except Exception as e:
            result = TestResult(test_name, "ERROR", 0.0,
                                f"Unhandled exception: {e}\n{traceback.format_exc()}")
        results.append(result)
        print(f"  => {result.status}: {result.detail} ({result.duration_s:.1f}s)")

    # Write JSON results
    json_path = f"{log_dir}/validation_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘", "ERROR": "⚠"}[r.status]
        print(f"  {icon} {r.name}: {r.status} — {r.detail}")
    passed = sum(1 for r in results if r.status == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    print(f"\n  Full results: {json_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
