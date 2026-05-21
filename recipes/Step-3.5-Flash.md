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
### FP8 on 8 GPUs (TP8 + FP8 KV Cache)
```bash
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  -tp 8 \
  --kv_cache_dtype fp8 \
  --trust-remote-code
```
### Expert Parallelism
Use this variant to measure the effect of expert parallelism on the same TP8 FP8 baseline.
```bash
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  -tp 8 \
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
Run the first benchmark pass with `TP=8` and FP8 KV cache only.
| Scenario              |                ISL |  OSL |           Concurrency | Purpose              |
| --------------------- | -----------------: | ---: | --------------------: | -------------------- |
| Standard decode sweep |               1024 | 1024 | 4, 8, 16, 32, 64, 128 | Throughput scaling   |
| Prefill-heavy         | 8192, 32768, 65536 |    1 |               1, 4, 8 | Long-context prefill |
| Balanced long context | 8192, 32768, 65536 | 1024 |           1, 4, 8, 16 | Serving behavior     |
| Decode-heavy          |               8192 | 8192 |               1, 4, 8 | Long generation      |
Run each scenario for these variants:
- FP8 TP8
- FP8 TP8 + expert parallelism
- FP8 TP8 + MTP
- FP8 TP8 + expert parallelism + MTP
## Performance Baseline
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
| Variant            |  ISL |  OSL | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | MTP Acceptance |
| ------------------ | ---: | ---: | ----------: | ----------: | ------------------------: | -----------------------: | -------------: | -------------: | -------------: |
| FP8 TP8            | 1024 | 1024 |           4 |          40 |                       TBD |                      TBD |            TBD |            TBD |            N/A |
| FP8 TP8 + EP       | 1024 | 1024 |           4 |          40 |                       TBD |                      TBD |            TBD |            TBD |            N/A |
| FP8 TP8 + MTP      | 1024 | 1024 |           4 |          40 |                       TBD |                      TBD |            TBD |            TBD |            TBD |
| FP8 TP8 + EP + MTP | 1024 | 1024 |           4 |          40 |                       TBD |                      TBD |            TBD |            TBD |            TBD |
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
- The model card advertises MTP-3, but supported runtime depth may differ. Do not report MTP-3 unless ATOM is actually running three speculative tokens or equivalent.
- MTP results must include acceptance statistics; output throughput alone can hide poor draft efficiency.
- Expert parallelism should be measured as its own variant because it changes MoE communication behavior.
- Long-context runs at 32K and 64K should be treated as stress tests until smaller workloads are stable.
