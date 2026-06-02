# Step 3.5 Flash Usage Guide
[Step 3.5 Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is a sparse Mixture-of-Experts model from StepFun. It has 196B total parameters, about 11B active parameters per token, a 256K context window, hybrid sliding-window/full attention, and native Multi-Token Prediction (MTP) support.

This recipe tracks the FP8 checkpoint on AMD MI325X (gfx942) with **TP=8** and production defaults (level 3, CUDAGraph). Expert parallelism (`--enable-expert-parallel`) is deferred — the MoRI EP warmup path hangs on some nodes.

## Environment Setup

Bootstrap a single `.venv` per node from the ATOM repo root. All dependencies install via pip/uv into the venv — no system ROCm shims or manual `ROCM_HOME` exports.

### Install stack

```bash
cd /home/AMD/jamesmit/apps/ATOM
rm -rf .venv
uv venv --python 3.12 .venv
source .venv/bin/activate

export ROCM_INDEX=https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/   # adjust for your GPU

# 1. ROCm SDK (TheRock pip packages)
uv pip install --index-url "$ROCM_INDEX" "rocm[libraries,devel]"

# 2. Expand devel headers/tools — required before AITER JIT
rocm-sdk init
rocm-sdk test          # must pass before continuing

# 3. PyTorch (same index)
uv pip install --index-url "$ROCM_INDEX" torch torchaudio torchvision

# 4. MoRI (expert-parallel comms — installed even when running TP-only)
uv pip install --pre amd-mori-nightly

# 5. AITER (GPU kernels) — pip from git, not an editable NFS checkout
uv pip install "git+https://github.com/ROCm/aiter@466ccb7d0#submodule=recursive"

# 6. ATOM (editable install)
uv pip install -e .

# 7. Workload tooling
uv pip install pytest "lm_eval[api]"
```

Verify:

```bash
python -c "
import torch; assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
import aiter
import atom
print('imports ok')
"
```

Download model weights once per node:

```bash
huggingface-cli download stepfun-ai/Step-3.5-Flash-FP8 \
  --local-dir /data/jamesmit/models/Step-3.5-Flash-FP8
```

## Launching Server

### FP8 on 8 GPUs (TP8 + FP8 KV Cache, production defaults)

Use the repo orchestration scripts with a clean venv on `PATH`:

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
export AITER_LOG_LEVEL=WARNING

bash scripts/stop_atom_server.sh

bash scripts/start_atom_server.sh \
  /data/jamesmit/models/Step-3.5-Flash-FP8 8 8000 \
  --kv_cache_dtype fp8 --trust-remote-code --max-model-len 16384 &

bash scripts/wait_server_ready.sh 8000 10 30 /home/AMD/jamesmit/ATOM_logs/atom_server.log
```

Equivalent direct launch:

```bash
python -m atom.entrypoints.openai_server \
  --model /data/jamesmit/models/Step-3.5-Flash-FP8 \
  -tp 8 \
  --kv_cache_dtype fp8 \
  --trust-remote-code \
  --max-model-len 16384
```

Startup typically takes ~3 min (includes CUDAGraph capture). Confirm with `rocm-smi --showmemuse` (VRAM > 0% on all 8 GPUs).

## Benchmarking

Use ATOM's serving benchmark with random prompts and `ignore_eos` so requested output length is measured consistently.

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=/data/jamesmit/models/Step-3.5-Flash-FP8 \
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
  --percentile-metrics="ttft,tpot,itl,e2el"
```

Or via script:

```bash
bash scripts/run_benchmark.sh \
  /data/jamesmit/models/Step-3.5-Flash-FP8 8000 1024 1024 4 10
```

### Benchmark Results

Run on the TP=8 production path above. Fill in rows as results are measured.

| Scenario              |  ISL |  OSL | Concurrency | Num prompts | Total tok/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2EL (ms) | Status    |
| --------------------- | ---: | ---: | ----------: | ----------: | ----------: | -----------: | -------------: | -------------: | -------------: | --------- |
| Smoke (production)    | 1024 | 1024 |           4 |           8 |         703 |          347 |            299 |           10.6 |           9900 | Validated |
| Standard decode sweep | 8192 | 1024 |           4 |          40 |        2956 |          329 |            364 |           11.5 |          10900 | Validated |
| Standard decode sweep | 8192 | 1024 |           8 |          80 |        5128 |          576 |            450 |           13.2 |          12600 | Validated |
| Standard decode sweep | 8192 | 1024 |          16 |         160 |        8087 |          896 |            561 |           16.7 |          15800 | Validated |
| Standard decode sweep | 8192 | 1024 |          32 |         320 |       11709 |         1309 |            799 |           23.0 |          22100 | Validated |
| Standard decode sweep | 8192 | 1024 |          64 |         640 |       15064 |         1671 |           1211 |           36.4 |          34800 | Validated |
| Standard decode sweep | 8192 | 1024 |         128 |        1280 |       18328 |         2030 |           1963 |           60.1 |          57300 | Validated |
| Decode-heavy          | 8192 | 8192 |           1 |          10 |         193 |           97 |            315 |           10.3 |          76600 | Validated |
| Decode-heavy          | 8192 | 8192 |           4 |          40 |         705 |          350 |            369 |           11.0 |          80500 | Validated |
| Decode-heavy          | 8192 | 8192 |           8 |          80 |        1333 |          670 |            438 |           11.7 |          86900 | Validated |

## Accuracy Validation

Run `lm_eval` against the same launch configuration used for benchmarks.

```bash
lm_eval \
  --model local-completions \
  --model_args model=/data/jamesmit/models/Step-3.5-Flash-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,max_length=8192 \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs "max_gen_toks=512,temperature=0,until=Question:"
```

### lm_eval Results

| Date       | ATOM commit | Topology   | Mode                 | Task        | Metric           |  Score | Notes             |
| ---------- | ----------- | ---------- | -------------------- | ----------- | ---------------- | -----: | ----------------- |
| 2026-06-02 | `ea34eac2`  | TP=8 no-EP | level 3 (production) | GSM8K-5shot | flexible-extract | 87.64% | MI325X, full 1319 |
| 2026-06-02 | `ea34eac2`  | TP=8 no-EP | level 3 (production) | GSM8K-5shot | strict-match     | 87.72% | MI325X, full 1319 |

## Caveats

- Keep AITER's `3rdparty/composable_kernel` submodule synced to AITER's pinned hash. A drifted CK submodule breaks varlen prefill JIT builds.
- Clear compile cache before server restart after code changes: `rm -rf ~/.cache/atom/*`
- Set `--max-num-batched-tokens` explicitly before long-context runs above 16K input tokens.
- MTP results should include acceptance statistics; do not report an MTP depth unless ATOM is actually running that depth.
- When switching GPU arch (gfx942 ↔ gfx950), rebuild the venv and clear Triton/Inductor/ATOM compile caches.
- Expert parallelism (`--enable-expert-parallel`) is deferred — MoRI EP warmup hangs on some nodes. TP=8 without EP is validated for both accuracy and production throughput on MI325X.
