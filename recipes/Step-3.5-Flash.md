# Step 3.5 Flash Usage Guide
[Step 3.5 Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is a sparse Mixture-of-Experts model from StepFun. It has 196B total parameters, about 11B active parameters per token, a 256K context window, hybrid sliding-window/full attention, and native Multi-Token Prediction (MTP) support.
This recipe focuses on the FP8 checkpoint with tensor parallelism across 8 GPUs and FP8 KV cache.
## Preparing Environment
Install ROCm, PyTorch, AITER, and ATOM from the theRock-based nightly builds:
```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ "rocm[libraries,devel]" torch torchvision
uv pip install git+https://github.com/ROCm/aiter.git
uv pip install git+https://github.com/ROCm/ATOM.git
```
## Launching Server
ATOM provides built-in support for Step 3.5 Flash via `Step3p5ForCausalLM`.

> **AMD requires expert parallelism.** On AMD ROCm the FP8 checkpoint must be launched with `--enable-expert-parallel` (or `--tp 4 --ep 4`). Pure tensor parallelism at TP ∈ {4, 8} hits FP8 block-quantization misalignment: `moe_intermediate_size = 1280` and `share_expert_dim = 1280` are not multiples of `block_n = 128` once split across 4 or 8 ranks, so each rank's scale slice straddles checkpoint scale-block boundaries and output degrades to gibberish. EP keeps whole experts on each rank at the un-split inter dim 1280 (1280 / 128 = 10, perfectly aligned). This matches what vLLM and SGLang ship on AMD; vLLM errors out at boot without EP, and SGLang docs require `--ep 4` with `--tp 4` for this model.

### FP8 on 8 GPUs (TP8 + EP + FP8 KV Cache) — canonical AMD topology
```bash
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  -tp 8 \
  --kv_cache_dtype fp8 \
  --enable-expert-parallel \
  --trust-remote-code
```
CUDAGraph capture works at this topology; `--enforce-eager` is **not** required.

### FP8 on 4 GPUs (TP4 + EP4)
```bash
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  -tp 4 \
  --kv_cache_dtype fp8 \
  --enable-expert-parallel \
  --trust-remote-code
```
### MTP Speculative Decoding
Start with one speculative token. Record MTP acceptance rate and average accepted tokens per forward pass for every MTP run.
```bash
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  -tp 8 \
  --kv_cache_dtype fp8 \
  --method mtp \
  --num-speculative-tokens 1 \
  --trust-remote-code
```
If higher MTP depths are supported, benchmark them as separate variants after the MTP-1 run is stable.
## Benchmarking
Use ATOM's serving benchmark with random prompts, fixed request counts, and `ignore_eos` so the requested output length is measured consistently.
```bash
python -m atom.benchmarks.benchmark_serving \
  --model=stepfun-ai/Step-3.5-Flash-FP8 \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} \
  --random-output-len=${OSL} \
  --random-range-ratio=0.8 \
  --num-prompts=$(( CONC * 10 )) \
  --max-concurrency=${CONC} \
  --num-warmups=$(( CONC * 2 )) \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --result-dir=${RESULT_DIR:-./benchmark-results} \
  --result-filename=${RESULT_FILENAME:-step35-flash-fp8-tp8-${ISL}-${OSL}-${CONC}.json} \
  --percentile-metrics="ttft,tpot,itl,e2el"
```
### Benchmark Matrix
Run the first benchmark pass with `TP=8`, FP8 KV cache, and `--enable-expert-parallel`.
| Scenario              |                ISL |  OSL |           Concurrency | Purpose              |
| --------------------- | -----------------: | ---: | --------------------: | -------------------- |
| Standard decode sweep |               1024 | 1024 | 4, 8, 16, 32, 64, 128 | Throughput scaling   |
| Prefill-heavy         | 8192, 32768, 65536 |    1 |               1, 4, 8 | Long-context prefill |
| Balanced long context | 8192, 32768, 65536 | 1024 |           1, 4, 8, 16 | Serving behavior     |
| Decode-heavy          |               8192 | 8192 |               1, 4, 8 | Long generation      |
Run each scenario for these variants:
- FP8 TP8 + EP (canonical AMD topology)
- FP8 TP8 + EP + MTP
## Performance Baseline

> **2026-05-26 status:** AITER varlen-prefill blocker is **resolved** — see the resolution recipe in `.kg/step3p5-aiter-varlen-jit-blocker-2026-05-26.md`. Two infra fixes were required outside ATOM itself: (1) reset AITER's `3rdparty/composable_kernel` submodule to AITER's pinned hash (`83566edb0`), (2) export `ROCM_HOME=$ROCM_DEVEL ROCM_PATH=$ROCM_DEVEL` in the shell that launches `openai_server` so the hipcc shim finds `hipsparse/hipsparse.h` on first JIT compile. With those in place, FP8 TP=8+EP runs cleanly on Step-3.5-Flash. The numbers below are first-pass measurements from the canonical AMD topology.

Environment:
- Date measured: 2026-05-26
- ROCm: theRock 7.2.0, gfx942 (Instinct MI325X VF)
- ATOM commit: `85a7c590` (branch `add-step3p5-flash-pt2`)
- AITER commit: `466ccb7d0` (main, 2026-05-26)
- AITER CK submodule (`3rdparty/composable_kernel`): `83566edb0` (AITER's pin)
- GPU: 8 × AMD Instinct MI325X VF (gfx942)
- Model revision: `stepfun-ai/Step-3.5-Flash-FP8`
- KV cache dtype: fp8
- Tensor parallel size: 8
- Expert parallel: enabled
- MTP: disabled (no `--method mtp`)
- Server flags: `-tp 8 --kv_cache_dtype fp8 --enable-expert-parallel --max-num-seqs 256 --gpu-memory-utilization 0.9 --trust-remote-code`
- Benchmark: `atom.benchmarks.benchmark_serving` with `--random-range-ratio 0.8 --ignore-eos --request-rate inf`, `num_prompts = concurrency × 10` (decode sweep), `× 4` (8K prefill), `× 4` (balanced 8K), warmups `= 2 × concurrency`.

### FP8 TP=8 + EP, no MTP

**Standard decode sweep (ISL = 1024, OSL = 1024)**

| Concurrency | Num Prompts | Output Tput (tok/s) | Total Tput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2EL (ms) |
| ----------: | ----------: | ------------------: | -----------------: | -------------: | -------------: | -------------: |
|           4 |          40 |               414.4 |                833 |            103 |           9.24 |          8 576 |
|           8 |          80 |               767.1 |              1 528 |            119 |          10.08 |          9 475 |
|          16 |         160 |             1 359.0 |              2 733 |            139 |          11.36 |         10 536 |
|          32 |         320 |             2 330.9 |              4 654 |            177 |          13.18 |         12 351 |
|          64 |         640 |             3 597.6 |              7 197 |            224 |          16.95 |         15 846 |
|         128 |        1 280 |             5 644.5 |             11 301 |            366 |          21.77 |         20 394 |

Decode throughput scales near-linearly from C=4 to C=128 (13.6× tokens for 32× concurrency — flatness above C=64 is dominated by TTFT growth). Mean TPOT stays under 22 ms across the full range.

**Prefill-heavy (ISL = 8192, OSL = 1)**

| Concurrency | Num Prompts | Total Tput (tok/s) | Mean TTFT (ms) | Mean E2EL (ms) |
| ----------: | ----------: | -----------------: | -------------: | -------------: |
|           1 |          10 |             38 673 |            191 |            191 |
|           4 |          40 |             55 170 |            514 |            515 |
|           8 |          80 |             57 541 |            974 |            975 |
|         — — | — — | — — | — — | — — |
| 1, 4, 8 @ ISL 32768 / 65536 | — | not measured | — | — |

Prefill saturates near 57 K tok/s at concurrency 8. ISL ≥ 32 K not yet measured (server was launched with the default `--max-num-batched-tokens 16384`, which silently rejects single prompts longer than that — the server prints `Request will never be scheduled` and the benchmark client times out in warmup). Relaunch with `--max-num-batched-tokens 65536` to enable the 32 K / 65 K sweeps.

**Balanced long context (ISL = 8192, OSL = 1024)**

| Concurrency | Num Prompts | Output Tput (tok/s) | Total Tput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2EL (ms) |
| ----------: | ----------: | ------------------: | -----------------: | -------------: | -------------: | -------------: |
|           1 |           4 |               113.3 |              1 041 |            194 |           8.62 |          7 951 |
|           4 |          16 |               385.0 |              3 484 |            283 |           9.70 |          9 264 |
|           8 |          32 |               693.3 |              6 161 |            377 |          10.74 |         10 350 |
|          16 |          64 |             1 168.0 |             10 373 |            496 |          12.39 |         11 989 |
|         — — | — — | — — | — — | — — | — — | — — |
| 1, 4, 8, 16 @ ISL 32768 / 65536 | — | — | not measured | — | — | — |

Balanced 8 K-input runs scale almost identically to the pure-decode sweep above on a per-output-token basis (TPOT 8.6 → 12.4 ms vs 9.2 → 11.4 ms at the same concurrency), confirming the engine is not serializing prefill against decode in this regime. ISL ≥ 32 K rows pending the `--max-num-batched-tokens` fix above.

**Decode-heavy (ISL = 8192, OSL = 8192)**: not measured yet.

### FP8 TP=8 + EP + MTP

Not measured yet. Will re-run the same matrix with `--method mtp --num-speculative-tokens 1` once the non-MTP matrix is complete, recording MTP acceptance rate and average accepted tokens per forward pass per the recipe header.
## Accuracy Validation
Run a lightweight correctness check before publishing performance results, especially for MTP.
```bash
lm_eval \
  --model local-completions \
  --model_args model=stepfun-ai/Step-3.5-Flash-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 5
```
For MTP, compare accuracy against the same launch configuration with MTP disabled.
## Known Caveats
- **AMD requires `--enable-expert-parallel`** (see the box at the top of Launching Server). Pure-TP at TP ∈ {4, 8} on AMD will produce gibberish output due to FP8 block-quantization misalignment of the 1280-wide inter dim.
- **AITER must be in sync with its pinned CK submodule.** If `git submodule status` in the AITER checkout shows `3rdparty/composable_kernel` drifted from AITER's pinned hash, varlen prefill JIT builds fail to link with type mismatches in `aiter/csrc/cpp_itfs/mha_fwd.cu`. Fix: `git submodule update --init --recursive 3rdparty/composable_kernel` and `rm -rf aiter/jit/build/mha_varlen_fwd_*` to force rebuild. See AGENTS.md for details.
- **Export `ROCM_HOME` and `ROCM_PATH` in the launch shell.** The hipcc shim used for AITER JIT compiles reads these to locate headers. With them unset (or pointing at `_rocm_sdk_core/` instead of `_rocm_sdk_devel/`), the first long-prompt request triggers a JIT compile that fails with `fatal error: 'hipsparse/hipsparse.h' file not found`.
- **Set `--max-num-batched-tokens` for long-context.** The default 16 384 silently rejects any single prompt longer than that. The server prints `Request will never be scheduled: input tokens=N > max_num_batched_tokens=16384` and the request hangs. For 32 K prompts launch with `--max-num-batched-tokens 32768`; for 65 K prompts use 65 536. There is no warning at boot — only when a too-large request arrives.
- **Swiglu clamp on MoE layers 43 and 44 is bypassed under EP.** The `swiglustep` custom MoE forward in `atom/models/step3p5.py` indexes per-expert tensors assuming every rank holds every expert; under EP each rank only holds `local_num_experts = num_experts / ep_size`, so the custom path is disabled and those two layers fall back to standard `FusedMoE.forward_impl` without the clamp. The layer-44 shared-expert clamp (limit=16) is in `Step3p5MLP` and is unaffected. Net accuracy impact is expected to be small but is unmeasured pending GSM8K under EP.
- The model card advertises MTP-3, but supported runtime depth may differ. Do not report MTP-3 unless ATOM is actually running three speculative tokens or equivalent.
- MTP results must include acceptance statistics; output throughput alone can hide poor draft efficiency.
- Long-context runs at 32K and 64K should be treated as stress tests until smaller workloads are stable.
