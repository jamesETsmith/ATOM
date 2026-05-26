# AGENTS.md

## Project overview

ATOM is a Python LLM inference engine for AMD ROCm, built around AITER kernels and exposed through an OpenAI-compatible server. The main execution path is:

`atom.entrypoints.openai_server` -> `atom.entrypoints.openai.api_server` -> `LLMEngine` -> `CoreManager` -> `EngineCore` -> `Scheduler` -> `ModelRunner`

The repository is not a generic web app. Most changes affect model execution, multi-process orchestration, or GPU/runtime behavior, so read the engine path before changing behavior that looks local.

## Before you change code

- Read `CLAUDE.md` first. It contains repository-specific rules that are stricter than what you can infer from source alone.
- Do not modify `@support_torch_compile`-decorated model files unless the task explicitly requires it and you have ruled out changing a call site instead. The local guidance warns that these edits can break Dynamo tracing even under eager execution.
- For multiprocessing changes, preserve the `spawn` start method. `CoreManager` explicitly sets it in `atom/model_engine/engine_core_mgr.py`.
- If you touch server startup, compilation, or CUDA-graph behavior, remember that stale compile cache under `/root/.cache/atom/*` can cause misleading runtime failures after code changes.

## Essential commands

## Environment and install

Use `uv` for environment management in this repository.

Verified local setup in this repo:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --extra-index-url https://rocm.nightlies.amd.com/v4/whl/ --pre "rocm[devel,device-gfx1100]"
uv pip install --python .venv/bin/python -e .
```

Notes from the verified setup:

- TheRock ROCm builds are `uv pip` installable.
- The command above successfully installed the multi-arch ROCm package set from theRock plus this project into `.venv`.
- The GPU target in theRock extras is hardware-specific. `device-gfx1100` was the verified target used here; switch it to the correct `device-gfx*` extra for the machine you are on.
- The theRock release docs also publish per-family indexes under `https://rocm.nightlies.amd.com/v2/...`; use those if you need a family-specific index instead of the multi-arch `v4/whl/` feed.

## Test

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest tests/
```

The editable install does not pull in `pytest`, so install it explicitly in the venv before running tests.

The unit suite under `tests/` is designed to run without a GPU by stubbing heavy imports such as `atom`, `atom.config`, `zmq`, and `xxhash` in `tests/conftest.py`. Prefer adding tests there when possible.

## Format and lint

```bash
uv pip install --python .venv/bin/python black ruff
.venv/bin/python -m black . && .venv/bin/python -m ruff check .
```

These commands are called out in `CLAUDE.md` as CI-enforced.

## Run the server

```bash
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8 -tp 8
```

## Run offline inference

```bash
python -m atom.examples.simple_inference --model <model> --kv_cache_dtype fp8
```

## Benchmark a running server

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=<model> --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random --random-input-len=1024 --random-output-len=1024 \
  --random-range-ratio=0.8 --num-prompts=1280 --max-concurrency=128 \
  --request-rate=inf --ignore-eos --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

## Profile

```bash
python -m atom.entrypoints.openai_server \
  --model <model> --kv_cache_dtype fp8 -tp 8 \
  --torch-profiler-dir ./trace --mark-trace

python tools/parse_trace.py ./trace/rank_0/<trace>.json.gz --layer 3
python tools/analyze_trace_summary.py ./trace/rank_0/<trace>.json.gz
```

## Accuracy eval

```bash
pip install lm-eval[api]
lm_eval --model local-completions \
  --model_args model=<model>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5
```

## High-value code locations

- `atom/entrypoints/openai_server.py`: stable CLI entry point; immediately delegates to the real API server module.
- `atom/model_engine/llm_engine.py`: public engine API, tokenizer loading, request preprocessing, fanout for `SamplingParams.n`, and result postprocessing.
- `atom/model_engine/engine_core_mgr.py`: launches engine processes, owns ZMQ sockets, waits for READY signals, and forces multiprocessing `spawn`.
- `atom/model_engine/engine_core.py`: per-process execution loop, scheduler integration, model-runner lifecycle, stream output handling, and KV-transfer coordination.
- `atom/model_engine/scheduler.py`: prefill-first batching, block allocation coordination, speculative decode stats, prefix cache stats, and postprocess behavior.
- `atom/model_engine/model_runner.py`: model loading, distributed init, CUDA-graph capture, forward pass execution, deferred output, and speculative decoding wiring.
- `atom/config.py`: master configuration dataclasses, compilation/cudagraph config, quantization config, and plugin-related config.
- `atom/utils/envs.py`: authoritative list of `ATOM_*` environment variables. Add new ATOM env vars here instead of scattering `os.getenv` calls.
- `tests/conftest.py`: the pattern for lightweight unit tests that avoid real GPU/HF/ZMQ initialization.

## Architecture and control flow

### Request lifecycle

Observed flow from docs and source:

1. User-facing APIs call `LLMEngine.add_request()` or `LLMEngine.generate()`.
2. `InputOutputProcessor.preprocess()` tokenizes input and creates `Sequence` objects.
3. `CoreManager.add_request()` serializes sequences and dispatches them to engine processes over ZMQ.
4. `EngineCore` input thread receives requests and feeds the scheduler.
5. `Scheduler.schedule()` chooses a prefill or decode batch, subject to sequence count, token budget, and KV block availability.
6. `ModelRunner.forward()` prepares metadata, runs the model, and postprocesses sampled tokens.
7. `Scheduler.postprocess()` mutates `Sequence` state, enforces stop conditions, and releases KV blocks for finished work.
8. Finished or streamed outputs go back through `EngineCore` -> `CoreManager` -> `InputOutputProcessor.postprocess()`.

### Process model

- `LLMEngine` is user-facing.
- `CoreManager` spawns one `EngineCore` process per DP rank.
- Each `EngineCore` creates an `AsyncIOProcManager`, which then spawns one worker per TP rank.
- IPC between manager and engine cores uses ZMQ `ROUTER`/`DEALER` for input and `PUSH`/`PULL` for output.
- READY signaling during startup matters: `EngineCore` starts its input thread before DP init so the manager can finish bringing up all ranks.

This means code that looks like a simple method call often crosses process boundaries. Be careful about picklability, queue contents, and message protocol changes.

### Scheduler behavior

The scheduler is intentionally prefill-first, not fairness-first. If you change queueing behavior, check both prompt throughput and decode latency implications.

It also coordinates:

- KV cache block allocation through `BlockManager`
- prefix caching statistics
- speculative decoding statistics
- KV transfer / P-D disaggregation metadata when enabled

### Forward-context pattern

The codebase uses a module-level global forward context (`atom/utils/forward_context.py`, described in `docs/architecture_guide.md`) to move attention metadata and KV references through CUDA-graph-compatible paths. Do not casually replace this with explicit parameter threading in hot paths without understanding the CUDA graph implications.

## Project conventions and patterns

### Configuration

- `Config` in `atom/config.py` is the canonical runtime config. Keyword args passed into `LLMEngine` are filtered against dataclass field names.
- `atom/utils/envs.py` is the single observed registry for ATOM-owned environment variables. The file explicitly documents that third-party env vars are documented there but not managed there.
- Plugin integration with vLLM is declared in `pyproject.toml` entry points and can be disabled with env vars documented in `atom/utils/envs.py`.

### Model registration

Supported HF architecture names are mapped to implementation class paths in `support_model_arch_dict` in `atom/model_engine/model_runner.py`. If you add model support, this registry is one of the required integration points.

### Testing style

- Many unit tests bypass heavy constructors with `__new__`, local stubs, and module injection rather than creating real engines.
- `tests/conftest.py` replaces the top-level `atom` package in `sys.modules` so `atom/__init__.py` does not trigger `LLMEngine` imports and heavy runtime setup.
- `MockConfig` in `tests/conftest.py` only provides attributes that the target unit reads. Follow that pattern instead of importing the real config when writing isolated tests.
- Sequence IDs are made deterministic in tests by resetting `Sequence.counter` with `itertools.count()`.

### Naming and maintenance guidance

`CLAUDE.md` contains two maintenance rules worth preserving in code changes:

- Fix the root pattern everywhere after finding one bug, not in just one file.
- Keep names aligned with behavior; rename stale identifiers when behavior changes.

## Node setup rules

- **Never cache in /home**: `/home` is NFS-mounted and extremely slow. Always set `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `TRITON_CACHE_DIR` to local storage (e.g., `/data/jamesmit/`). Delete any HF cache that lands in `~/.cache/huggingface/` immediately.
- **Use /data for models and caches**: On multi-node AMD clusters, `/data` is typically local NVMe. Create `/data/jamesmit/models/` and `/data/jamesmit/hf_cache/` on each new node before running anything.
- **Download models explicitly**: When switching nodes, the model checkpoint won't exist on the new node. Download it to `/data/jamesmit/models/` before launching the server. Don't assume paths from other nodes exist.
- **Run `rocm-sdk init` in every new venv**: After installing `rocm[devel]`, run `.venv/bin/rocm-sdk init` to expand the devel SDK contents (headers, tools). Without this, AITER JIT compilation fails with missing `thrust/complex.h` and `hipsparse/hipsparse.h`. This is required for v2 per-family indexes; v4 may self-extract.
- **AITER JIT fix**: Each new venv needs a `clang++.cfg` file next to its clang++ binary containing `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13`. The clang++ location varies by ROCm package version — after `rocm-sdk init`, check both `_rocm_sdk_devel/lib/llvm/bin/` and `_rocm_sdk_core/lib/llvm/bin/`; create the cfg in whichever one `hipcc` actually uses. May need it in both.
- **Clear compile caches when switching GPU arch**: Triton cache (`~/.triton/cache/`), torch inductor cache (`/tmp/torchinductor_*/`), and ATOM compile cache (`~/.cache/atom/torch_compile_cache/`) contain GPU-arch-specific compiled kernels. When switching between gfx942 and gfx950 nodes, always `rm -rf` all three before launching the server. Symptom of stale cache: `hipErrorInvalidDeviceFunction` during warmup.
- **AITER JIT build cache is in-tree and NFS-shared**: AITER stores compiled `.so` files in `aiter/jit/*.so` and `aiter/jit/build/`. Since the aiter repo is on NFS, these are shared across nodes. When switching GPU arch, run `rm -rf /path/to/aiter/aiter/jit/*.so /path/to/aiter/aiter/jit/build/` and also delete any `.hsaco` files under `aiter/ops/triton/configs/`. First launch after cleaning will trigger JIT recompilation (~3-5 min).
- **theRock pip indexes**: Use per-family v2 indexes for GPU-specific installs: `https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/` for MI300/MI325 (gfx942), `https://rocm.nightlies.amd.com/v2/gfx95X-dcgpu/` for MI350 (gfx950). The v4 multi-arch index also works with `rocm-sdk-device-gfx942` etc.
- **Set `ROCM_HOME` to `_rocm_sdk_devel`**: After `rocm-sdk init`, the devel SDK is expanded into `_rocm_sdk_devel/`. Set `ROCM_HOME` and `ROCM_PATH` to this directory so AITER JIT compilation finds headers. Keep `HIP_PATH` pointing to `_rocm_sdk_core/`.
- **`libamdhip64.so` symlink**: Some theRock builds have `libamdhip64.so.7` but no `libamdhip64.so` unversioned symlink. Create it in `_rocm_sdk_core/lib/` if missing: `ln -s libamdhip64.so.7 libamdhip64.so`.

## Non-obvious gotchas

- `atom/__init__.py` is heavy enough that tests intentionally avoid importing it. If a new test imports `atom` directly, expect unrelated initialization pain.
- `InputOutputProcessor.preprocess()` intentionally rejects `SamplingParams.n > 1`; multi-choice fanout must go through `preprocess_fanout()`. There are dedicated tests for this in `tests/test_io_processor_fanout.py`.
- Fanout siblings are marked `needs_independent_noise=True` so identical logits do not produce identical samples. If you touch sampling or batch metadata, preserve that behavior.
- Deferred sampled-token output in `tokenIDProcessor` is disabled when `kv_transfer_config` is set; P/D disaggregation follows different output constraints.
- `CoreManager` mutates parallel settings when `enable_dp_attention` is enabled: it expands local engine count and collapses TP to 1. Changes around parallel config should account for that rewrite.
- `EngineCore` sends a READY sentinel before normal work begins. Startup changes must preserve the READY/SHUTDOWN protocol expected by `CoreManager._wait_for_all_ready_signals()`.
- `atom/utils/envs.py` uses lazy `__getattr__` evaluation. Code should access envs as attributes on `atom.utils.envs`, not by assuming constants were precomputed.

## CI and repo signals worth knowing

- The primary CI workflow is `.github/workflows/atom-test.yaml`.
- PRs that only touch markdown/docs/license/gitignore are ignored by that workflow trigger.
- Accuracy CI is GPU/container heavy and downloads or resolves an AITER wheel before running model tests; local unit tests are much lighter and should be your first validation layer.

## Additional docs

Use these when the task touches their area:

- `docs/architecture_guide.md`
- `docs/configuration_guide.md`
- `docs/compilation_cudagraph_guide.md`
- `docs/distributed_guide.md`
- `docs/environment_variables.md`
- `docs/model_ops_guide.md`
- `docs/model_support_guide.md`
- `docs/scheduling_kv_cache_guide.md`
- `docs/serving_benchmarking_guide.md`

For model- or deployment-specific behavior, the `recipes/` directory is substantive and often contains the real expected launch shape for a supported model family.
