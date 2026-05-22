---
name: step3p5-ep-dispatch-bug
type: decision
tags: [step3p5, ep, mori, bug, fix]
created: 2026-05-22
updated: 2026-05-22
status: active
---

# EP Dispatch Bug: fused_experts is None with dp_size=1

## Summary
ATOM's EP mode crashed because `fused_experts` was None when `dp_size=1`. Two bugs were found and fixed: (1) `use_all2all_kernels` incorrectly gated on `dp_size > 1`, blocking MoRI creation for pure EP, and (2) token truncation in modular_kernel.py and step3p5.py used `dp_size` instead of `num_dispatchers`, discarding 7/8 of dispatched tokens.

## Details

### Root Cause Chain (Pre-Fix)

1. User runs with `--enable-expert-parallel -tp 8` (no DP)
2. `FusedMoEParallelConfig.make()` sets `ep_size=8, dp_size=1, use_ep=True` (line 134 of `moe.py`)
3. `use_all2all_kernels` returned `self.dp_size > 1 and self.use_ep and _has_module("mori")` — **False** because `dp_size=1`
4. `fused_experts` stayed `None`
5. Swiglustep EP path crashed: `self.experts.quant_method.fused_experts._prepare()` → `AttributeError: 'NoneType'`

### Fixes Applied (commit 7ba0a3a)

**Bug 1 — `use_all2all_kernels` gate** (`atom/model_ops/moe.py:82`):
```python
# Before (broken for pure EP):
return self.dp_size > 1 and self.use_ep and _has_module("mori")
# After:
return self.use_ep and _has_module("mori")
```
MoRI dispatch/combine operates over the EP group communicator (`get_ep_group().device_communicator.all2all_manager`) and has no dependency on `dp_size`. The gate was an artificial restriction.

**Bug 2 — Token truncation multiplier** (`atom/model_ops/fused_moe/modular_kernel.py:336` and `atom/models/step3p5.py:311`):
```python
# Before (discards 7/8 of tokens with pure EP):
dp_size = get_dp_group().world_size  # = 1
total_valid_tokens = context.graph_bs * topk * dp_size
# After:
num_dispatchers = self.prepare_finalize.num_dispatchers()  # = 8
total_valid_tokens = context.graph_bs * topk * num_dispatchers
```
`num_dispatchers` comes from `all2all_manager.world_size` (set during MoRI construction), which equals `ep_size` in pure EP mode.

### DP Attention Workaround — Why It Failed

We also tried `--enable-dp-attention --enable-expert-parallel -tp 8`, expecting CoreManager to set `dp_size=8` per process. It crashed with the same error because DP attention creates 8 separate OS processes (one per GPU), each with its own process group. Within each process, `get_dp_group().world_size` was still 1. The fix to `use_all2all_kernels` resolves this for both approaches.

### Pre-Existing Issue: expert_mask vs expert_map

`forward_impl()` (moe.py:2789) passes `expert_map=self.expert_mask` — a boolean mask where an expert_map (global→local index mapping) is expected. This is a separate bug being addressed by PRs #875 and #887 on main. Our EP fix enables the MoRI path, which handles dispatch internally via topk_ids and doesn't rely on expert_map during dispatch. The local `fused_moe()` computation does receive expert_map, but this is a pre-existing issue and not introduced by our changes.

### GPU Validation Status

Code fixes committed (7ba0a3a) and pushed to fork. GPU validation of EP is **BLOCKED** because the `mori` Python package is not installed on the test node (quanta-ccs-aus-f01-40). MoRI is required for `_has_module("mori")` to return True, which is the remaining gate in `use_all2all_kernels`. The `dp_size > 1` gate removal is confirmed correct via debug logging — `init_prepare_finalize` is called for all FusedMoE layers, but returns None because `_has_module("mori")` is False.

To validate: install MoRI package (AMD internal, requires MPI + CMake + hipcc build), then run `python tools/validate_ep_fixes.py --model <path>`.

Baseline TP=8 and MTP TP=8 validated successfully (5/5 first-word match).

## Relationships
- depends-on: [[step3p5-mtp-ep-fix-2026-05-22]]
- used-by: [[step3p5-gpu-validation-2026-05-22]]
- related: [[step3p5-flash]]
