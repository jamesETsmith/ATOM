#!/usr/bin/env python3
"""Verify Step-3.5-Flash MTP startup and basic inference.

Usage:
    python tools/verify_step3p5_mtp.py \
        --model /data/jamesmit/models/Step-3.5-Flash-FP8 \
        --tp 8

This script:
  1. Starts an ATOM server with MTP enabled (--method mtp --num-speculative-tokens 1)
  2. Sends a few test prompts
  3. Reports MTP acceptance rate and output quality
  4. Compares output with and without MTP (if --compare flag is set)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    """Poll /health until the server is ready."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    return False


def send_completion(base_url: str, prompt: str, max_tokens: int = 128) -> dict:
    """Send a completion request and return the response."""
    import urllib.request

    payload = json.dumps({
        "model": "test",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--compare", action="store_true",
                        help="Also run without MTP and compare outputs")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a Python function that computes fibonacci numbers.",
        "The meaning of life is",
        "List 5 benefits of exercise:",
    ]

    def run_server(mtp: bool) -> subprocess.Popen:
        cmd = [
            sys.executable, "-m", "atom.entrypoints.openai_server",
            "--model", args.model,
            "-tp", str(args.tp),
            "--kv_cache_dtype", "fp8",
            "--port", str(args.port),
            "--trust-remote-code",
        ]
        if mtp:
            cmd += ["--method", "mtp",
                    "--num-speculative-tokens", str(args.num_speculative_tokens)]
        env = os.environ.copy()
        env["AITER_LOG_LEVEL"] = "WARNING"
        proc = subprocess.Popen(cmd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc

    def kill_server(proc):
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Run with MTP
    print(f"Starting server with MTP (spec_tokens={args.num_speculative_tokens})...")
    proc = run_server(mtp=True)
    try:
        if not wait_for_server(base_url, timeout=args.timeout):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            print(f"FAIL: Server did not start within {args.timeout}s")
            print(f"stderr tail:\n{stderr[-2000:]}")
            proc.kill()
            sys.exit(1)

        print("Server is ready. Sending test prompts...")
        mtp_results = []
        for i, prompt in enumerate(test_prompts):
            try:
                resp = send_completion(base_url, prompt)
                text = resp["choices"][0]["text"].strip()
                usage = resp.get("usage", {})
                mtp_results.append({
                    "prompt": prompt,
                    "output": text[:200],
                    "completion_tokens": usage.get("completion_tokens", -1),
                })
                print(f"  [{i+1}/{len(test_prompts)}] {prompt[:40]}... -> {text[:80]}...")
            except Exception as e:
                print(f"  [{i+1}/{len(test_prompts)}] ERROR: {e}")
                mtp_results.append({"prompt": prompt, "error": str(e)})

        print(f"\nMTP Results: {len([r for r in mtp_results if 'error' not in r])}/{len(test_prompts)} prompts succeeded")
    finally:
        kill_server(proc)

    # Optionally compare without MTP
    if args.compare:
        print(f"\nStarting server WITHOUT MTP for comparison...")
        proc = run_server(mtp=False)
        try:
            if not wait_for_server(base_url, timeout=args.timeout):
                print("FAIL: Non-MTP server did not start")
                proc.kill()
                sys.exit(1)

            print("Sending same prompts without MTP...")
            for i, prompt in enumerate(test_prompts):
                try:
                    resp = send_completion(base_url, prompt)
                    text = resp["choices"][0]["text"].strip()
                    mtp_text = mtp_results[i].get("output", "")
                    match = "MATCH" if text[:50] == mtp_text[:50] else "DIFFER"
                    print(f"  [{i+1}] {match}: {text[:80]}...")
                except Exception as e:
                    print(f"  [{i+1}] ERROR: {e}")
        finally:
            kill_server(proc)

    print("\nDone.")


if __name__ == "__main__":
    main()
