#!/usr/bin/env python3
"""Focused Step-3.5-Flash validation for add-step3p5-flash branch.

Tests:
  1. Baseline TP=8 - server starts, produces output
  2. MTP TP=8 - server starts, produces output, compare vs baseline
  3. EP TP=8 - attempt, expected to fail without MoRI

Usage:
    .venv/bin/python tools/validate_ep_fixes.py \
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

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a Python function to compute the nth Fibonacci number.",
    "The meaning of life is",
    "List 5 benefits of regular exercise:",
]


@dataclass
class TestResult:
    name: str
    status: str
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
        "--kv_cache_dtype", "fp8", "--server-port", str(port),
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
                        max_tokens: int = 128,
                        model: str = "test") -> list[dict]:
    results = []
    for p in prompts:
        try:
            resp = send_completion(base_url, p, max_tokens, model=model)
            text = resp["choices"][0]["text"].strip()
            results.append({"prompt": p, "output": text,
                           "tokens": resp.get("usage", {}).get("completion_tokens", -1)})
        except Exception as e:
            results.append({"prompt": p, "error": str(e)})
    return results


def test_baseline(model: str, log_dir: str) -> tuple[TestResult, list[dict]]:
    """Test 1: Baseline TP=8."""
    port = PORT_BASE
    log = f"{log_dir}/baseline_tp8.log"
    t0 = time.time()
    proc = start_server(model, 8, port, [], log)
    try:
        ok = wait_for_server(f"http://localhost:{port}", timeout=600)
        dur = time.time() - t0
        if not ok:
            tail = Path(log).read_text()[-2000:]
            return TestResult("baseline_tp8", "FAIL", dur,
                              f"Server did not start\n{tail}"), []
        results = collect_completions(f"http://localhost:{port}", PROMPTS,
                                       model=model)
        errors = [r for r in results if "error" in r]
        return TestResult("baseline_tp8", "PASS" if not errors else "FAIL",
                          dur, f"{len(results)-len(errors)}/{len(results)} prompts OK",
                          {"completions": results}), results
    finally:
        kill_server(proc)


def test_mtp(model: str, log_dir: str, baseline_results: list[dict]) -> TestResult:
    """Test 2: MTP TP=8 + first-word comparison."""
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
            return TestResult("mtp_tp8", "FAIL", dur,
                              f"MTP server did not start\n{tail}")
        mtp_results = collect_completions(f"http://localhost:{port}", PROMPTS,
                                             model=model)
        errors = [r for r in mtp_results if "error" in r]
        if errors:
            return TestResult("mtp_tp8", "FAIL", dur,
                              f"{len(errors)} prompts failed",
                              {"completions": mtp_results})

        # Compare first words
        matches = 0
        total = 0
        comparisons = []
        for b, m in zip(baseline_results, mtp_results):
            if "error" in b or "error" in m:
                continue
            total += 1
            b_first = b["output"].split()[0] if b["output"] else ""
            m_first = m["output"].split()[0] if m["output"] else ""
            matched = b_first == m_first
            if matched:
                matches += 1
            comparisons.append({
                "prompt": b["prompt"][:50],
                "base_first": b_first[:30],
                "mtp_first": m_first[:30],
                "match": matched,
            })

        ratio = f"{matches}/{total}"
        status = "PASS" if total > 0 and matches / total >= 0.6 else "FAIL"
        return TestResult("mtp_tp8", status, dur,
                          f"First-word match: {ratio}",
                          {"comparisons": comparisons, "completions": mtp_results})
    finally:
        kill_server(proc)


def test_ep(model: str, log_dir: str) -> TestResult:
    """Test 3: EP TP=8 (--enable-expert-parallel)."""
    port = PORT_BASE + 2
    log = f"{log_dir}/ep_tp8.log"
    t0 = time.time()
    proc = start_server(model, 8, port, ["--enable-expert-parallel"], log)
    try:
        ok = wait_for_server(f"http://localhost:{port}", timeout=600)
        dur = time.time() - t0
        if not ok:
            tail = Path(log).read_text()[-3000:]
            # Check for known blockers
            if "mori" in tail.lower() or "use_all2all" in tail.lower():
                return TestResult("ep_tp8", "BLOCKED", dur,
                                  "MoRI package not installed — EP requires mori for all-to-all dispatch",
                                  {"log_tail": tail})
            return TestResult("ep_tp8", "FAIL", dur,
                              f"EP server did not start\n{tail}")
        results = collect_completions(f"http://localhost:{port}", PROMPTS[:3],
                                       model=model)
        errors = [r for r in results if "error" in r]
        return TestResult("ep_tp8", "PASS" if not errors else "FAIL",
                          dur, f"{len(results)-len(errors)}/{len(results)} prompts OK",
                          {"completions": results})
    finally:
        kill_server(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--log-dir", default="./validation_logs/ep_fixes")
    parser.add_argument("--skip-ep", action="store_true",
                        help="Skip EP test (known blocked without MoRI)")
    args = parser.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Clear compile caches
    import shutil
    for p in [os.path.expanduser("~/.cache/atom/torch_compile_cache")]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
            os.makedirs(p, exist_ok=True)

    results: list[TestResult] = []

    # Test 1: Baseline
    print(f"\n{'='*60}\nTest 1: Baseline TP=8\n{'='*60}")
    t1, baseline_completions = test_baseline(args.model, log_dir)
    results.append(t1)
    print(f"  => {t1.status}: {t1.detail} ({t1.duration_s:.1f}s)")

    # Test 2: MTP
    print(f"\n{'='*60}\nTest 2: MTP TP=8\n{'='*60}")
    t2 = test_mtp(args.model, log_dir, baseline_completions)
    results.append(t2)
    print(f"  => {t2.status}: {t2.detail} ({t2.duration_s:.1f}s)")

    # Test 3: EP
    if not args.skip_ep:
        print(f"\n{'='*60}\nTest 3: EP TP=8\n{'='*60}")
        t3 = test_ep(args.model, log_dir)
        results.append(t3)
        print(f"  => {t3.status}: {t3.detail} ({t3.duration_s:.1f}s)")
    else:
        results.append(TestResult("ep_tp8", "SKIP", 0,
                                  "Skipped — MoRI not available"))

    # Check MoRI availability
    import importlib.util
    mori_available = importlib.util.find_spec("mori") is not None

    # Write JSON results
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "branch": "add-step3p5-flash",
        "mori_available": mori_available,
        "node": os.uname().nodename,
        "results": [asdict(r) for r in results],
    }
    json_path = f"{log_dir}/validation_results.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  MoRI available: {mori_available}")
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘",
                "ERROR": "⚠", "BLOCKED": "⊗"}[r.status]
        print(f"  {icon} {r.name}: {r.status} — {r.detail}")
    passed = sum(1 for r in results if r.status == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    print(f"  Full results: {json_path}")

    return 0 if all(r.status in ("PASS", "SKIP", "BLOCKED") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
