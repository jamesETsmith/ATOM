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
> **2026-05-26 status:** Server launches cleanly at `--tp 8 --enable-expert-parallel` with CUDAGraph enabled. A 4-prompt smoke test (capitals, arithmetic, story, QA) returns coherent output. The full benchmark matrix is currently **blocked on an AITER varlen-prefill issue unrelated to ATOM**: with `ENABLE_CK=1` the JIT build of `mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale` fails at link with CK API type mismatches in `aiter/csrc/cpp_itfs/mha_fwd.cu`, and with `ENABLE_CK=0` the Triton dispatcher rejects Step 3.5's sliding-window prefill (`window_size_right=0 ≠ -1`). Short prompts succeed because they hit a different pre-built varlen kernel. Re-run the matrix once AITER ships a CK snapshot whose API matches the bundled kernel stubs, or once the Triton path accepts causal SWA with `window_size_right=0`.

Fill this table after collecting the first validated numbers.
Environment:
- Date measured: TBD
- ROCm/theRock build: TBD
- ATOM commit: TBD
- AITER commit/version: TBD
- GPU: TBD
- Model revision: TBD
- KV cache dtype: fp8
- Tensor parallel size: 8
- Expert parallel: enabled
| Variant            |  ISL |  OSL | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | MTP Acceptance |
| ------------------ | ---: | ---: | ----------: | ----------: | ------------------------: | -----------------------: | -------------: | -------------: | -------------: |
| FP8 TP8 + EP       | 1024 | 1024 |           4 |          40 |                   blocked |                  blocked |        blocked |        blocked |            N/A |
| FP8 TP8 + EP + MTP | 1024 | 1024 |           4 |          40 |                   blocked |                  blocked |        blocked |        blocked |        blocked |
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
- **AITER varlen-prefill blocker (2026-05-26):** the current AITER snapshot cannot satisfy Step 3.5's varlen bf16 prefill with sliding-window attention on either backend (CK build fails, Triton rejects `window_size_right=0`). Short prompts work; benchmark and `lm_eval` long-prompt runs do not. This is upstream AITER, not ATOM.
- **Swiglu clamp on MoE layers 43 and 44 is bypassed under EP.** The `swiglustep` custom MoE forward in `atom/models/step3p5.py` indexes per-expert tensors assuming every rank holds every expert; under EP each rank only holds `local_num_experts = num_experts / ep_size`, so the custom path is disabled and those two layers fall back to standard `FusedMoE.forward_impl` without the clamp. The layer-44 shared-expert clamp (limit=16) is in `Step3p5MLP` and is unaffected. Net accuracy impact is expected to be small but is unmeasured pending GSM8K under EP.
- The model card advertises MTP-3, but supported runtime depth may differ. Do not report MTP-3 unless ATOM is actually running three speculative tokens or equivalent.
- MTP results must include acceptance statistics; output throughput alone can hide poor draft efficiency.
- Long-context runs at 32K and 64K should be treated as stress tests until smaller workloads are stable.
