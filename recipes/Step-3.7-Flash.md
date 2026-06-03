# Step 3.7 Flash Usage Guide

[Step 3.7 Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) is a sparse MoE vision-language model from StepFun. This recipe covers **text-only** inference on AMD via native ATOM: the language backbone matches Step 3.5 Flash (`step3p5` in HF `text_config`). Vision weights are skipped; image inputs are not supported on `openai_server` yet.

Checkpoint: `stepfun-ai/Step-3.7-Flash-FP8` on MI325X/MI355X with **TP=8**, FP8 KV cache, production defaults (level 3, CUDAGraph). Expert parallelism is not used.

## Environment Setup

Bootstrap a single `.venv` per node from the ATOM repo root. Commands below are intended to be copy-pasted verbatim.

### Install stack

```bash
cd /home/AMD/jamesmit/apps/ATOM
rm -rf .venv
uv venv --python 3.12 .venv
source .venv/bin/activate

export ROCM_INDEX=https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/
# MI350 (gfx950): export ROCM_INDEX=https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/

uv pip install --index-url "$ROCM_INDEX" "rocm[libraries,devel]"
rocm-sdk init
rocm-sdk test

uv pip install --index-url "$ROCM_INDEX" torch torchaudio torchvision
uv pip install --pre amd-mori-nightly
uv pip install "git+https://github.com/ROCm/aiter@466ccb7d0#submodule=recursive"
uv pip install -e .
uv pip install pytest "lm_eval[api]"
```

Verify:

```bash
python -c "
import torch; assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
import aiter
import atom.model_config.step3p5
print('imports ok')
"
```

Download weights once per node:

```bash
hf download stepfun-ai/Step-3.7-Flash-FP8 \
  --local-dir /data/jamesmit/models/Step-3.7-Flash-FP8
```

## Launching Server

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
export AITER_LOG_LEVEL=WARNING
export ROCM_DEVEL="$(realpath .venv/lib/python3.12/site-packages/_rocm_sdk_devel)"
export ROCM_HOME="$ROCM_DEVEL"
export ROCM_PATH="$ROCM_DEVEL"

python -m atom.entrypoints.openai_server \
  --model /data/jamesmit/models/Step-3.7-Flash-FP8 \
  -tp 8 \
  --kv_cache_dtype fp8 \
  --trust-remote-code \
  --max-model-len 16384
```

Startup typically takes ~3 minutes (includes CUDAGraph capture). Confirm with `rocm-smi --showmemuse` (VRAM > 0% on all 8 GPUs).

## Benchmarking

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=/data/jamesmit/models/Step-3.7-Flash-FP8 \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 \
  --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=40 \
  --max-concurrency=4 \
  --num-warmups=8 \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

### Benchmark Results

| Scenario | ISL | OSL | Concurrency | Num prompts | Total tok/s | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2EL (ms) | Status |
| -------- | --: | --: | ----------: | ----------: | ----------: | -----------: | -------------: | -------------: | -------------: | ------ |
| Smoke    | 1024 | 1024 | 4 | 40 | 695 | 346 | 201 | 11.0 | 10291 | Verified 2026-06-02 |

## Accuracy Validation

```bash
lm_eval \
  --model local-completions \
  --model_args model=/data/jamesmit/models/Step-3.7-Flash-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,max_length=8192 \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs "max_gen_toks=512,temperature=0,until=Question:"
```

### lm_eval Results

| Date | ATOM commit | Topology | Task | Metric | Score | Notes |
| ---- | ----------- | -------- | ---- | ------ | ----: | ----- |
| 2026-06-02 | feat/step-3.7-flash | TP=8 no-EP | GSM8K-5shot | flexible-extract | 88.78% | MI325X gfx942 |

## Caveats

- Text-only: `vision_model.*` and `vit_large_projector.*` weights are not loaded.
- Clear compile cache after code changes: `rm -rf ~/.cache/atom/*`
- Keep AITER CK submodule pinned; drift breaks varlen prefill JIT.
- MTP-3 speculative decoding is not enabled in ATOM yet.
- Multimodal serving (images) is deferred; see Kimi-K2.5 vLLM plugin pattern for future work.

## Verified

| Field | Value |
| ----- | ----- |
| Recipe verified | 2026-06-02 |
| GPU | MI325X (gfx942), 8× GPU |
| ATOM commit | feat/step-3.7-flash (post-5cf48f64: MoE FP8 block-scale TP shard fix) |
| Required exports | `ROCM_HOME`/`ROCM_PATH` → `_rocm_sdk_devel` if AITER JIT fails |
