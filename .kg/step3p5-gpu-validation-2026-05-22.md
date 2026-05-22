---
name: step3p5-gpu-validation-2026-05-22
type: process
tags: [step3p5, validation, gpu, mtp, ep]
created: 2026-05-22
updated: 2026-05-22
status: active
---

# Step-3.5-Flash GPU Validation Results (2026-05-22)

## Summary
Validated Step-3.5-Flash-FP8 on 8x MI325X (quanta-ccs-aus-f01-40). Baseline TP=8 and MTP TP=8 both PASS with 5/5 first-word match. EP TP=8 could not be validated because the `mori` package is not installed on this node — EP requires MoRI for all-to-all expert dispatch. The two EP code fixes (use_all2all_kernels gate + token truncation) are committed but untested on GPU.

## Details

### Test Results

| Test | Status | Duration | Detail |
|------|--------|----------|--------|
| Baseline TP=8 | PASS | 155s | 5/5 prompts OK |
| MTP TP=8 | PASS | 165s | First-word match: 5/5 |
| EP TP=8 | SKIP | - | MoRI not installed on node |

### MTP First-Word Comparison

| Prompt | Baseline | MTP | Match |
|--------|----------|-----|-------|
| What is the capital of France? | \|Paris | \|Paris | YES |
| Explain quantum computing... | - | - | YES |
| Write a Python function... | The | The | YES |
| The meaning of life is | a | a | YES |
| List 5 benefits of exercise | 1. | 1. | YES |

### MTP Output Quality

MTP produces coherent, complete outputs. One notable difference: the Fibonacci prompt gets a `</think>` tag in MTP output (Step-3.5-Flash uses a reasoning format), suggesting slightly different generation paths but semantically equivalent output.

### EP Blocker: MoRI Not Installed

- `import mori` fails — package not on PyPI or ROCm nightlies
- Source at `/home/AMD/lirzhang/mori` exists but binary incompatible (`undefined symbol: _ZN3c103hip19getCurrentHIPStreamEa`)
- Building from source requires MPI (`libmpi-dev`) which is not installed
- The `_has_module("mori")` check in `use_all2all_kernels` returns False, preventing FusedMoEModularKernel creation
- Without MoRI, `fused_experts` stays None for ALL FusedMoE layers when EP is enabled
- This blocks both swiglustep (crash) and standard MoE layers (silent wrong results via fallback to local-only `rocm_asm_moe_impl`)

### What Was Validated by Code Analysis (Not GPU)

1. **Bug 1 fix** (`moe.py:82`): Removed `dp_size > 1` gate from `use_all2all_kernels`. Confirmed with debug logging that `init_prepare_finalize` is called for all 45 FusedMoE layers with `use_all2all=False` (because `_has_module("mori")` returns False). With MoRI installed, this would return True.

2. **Bug 2 fix** (`modular_kernel.py:336`, `step3p5.py:311`): Changed `get_dp_group().world_size` to `prepare_finalize.num_dispatchers()`. Verified `num_dispatchers` = `all2all_manager.world_size` = ep_size during MoRI construction.

### Environment

- Node: quanta-ccs-aus-f01-40.adc.amd.com
- GPUs: 8x MI325X (gfx942), 256 GiB VRAM each
- Model: Step-3.5-Flash-FP8 (196B params, 288 experts, top-8)
- ATOM branch: add-step3p5-flash (commit 7ba0a3a)
- Python: 3.12, PyTorch: 2.8+ (theRock build)
- MoRI: NOT AVAILABLE

### To Validate EP on a Properly Equipped Node

1. Install MoRI: `pip install mori` (from AMD internal index or build from source at `/home/AMD/lirzhang/mori`)
2. Requires: MPI (libopenmpi-dev), CMake, hipcc
3. Run: `python tools/validate_ep_fixes.py --model /data/jamesmit/models/Step-3.5-Flash-FP8`
4. Expected: EP TP=8 server starts, produces correct output with all-to-all dispatch

### Results File

Full JSON: `validation_logs/ep_fixes/validation_results.json`

## Relationships
- depends-on: [[step3p5-ep-dispatch-bug]]
- depends-on: [[step3p5-mtp-ep-fix-2026-05-22]]
- related: [[step3p5-flash]]
