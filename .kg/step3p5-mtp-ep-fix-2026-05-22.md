---
name: step3p5-mtp-ep-fix-2026-05-22
type: decision
tags: [step3p5, mtp, ep, swiglustep, memory, fix]
created: 2026-05-22
updated: 2026-05-22
status: active
---

# Step-3.5-Flash MTP & EP Fixes (2026-05-22)

## Summary
Two fixes applied to unblock MTP startup and EP correctness for Step-3.5-Flash. MTP was OOMing during warmup due to pre-materialised bf16 expert weight buffers; EP was crashing due to global→local expert ID mismatch in the swiglustep unfused path.

## Details

### Fix 1: Deferred swiglustep bf16 materialisation (commit 53bd575)
- **Problem:** `_prepare_swiglustep_weights()` pre-materialised full `[E, 2*inter_pad, hidden]` bf16 buffers for swiglustep layers 43-44 (~8.4 GiB per layer, ~16.9 GiB total on TP=1). With MTP draft weights also loaded, total memory exceeded GPU capacity during warmup.
- **Solution:** Split into two phases:
  - Warmup/prefill: on-the-fly per-expert FP8→bf16 dequant via `_dequant_expert()` (~20 MiB peak)
  - Graph capture: lazily materialise bf16 buffers on first captured forward (after KV cache is already sized)
- **Impact:** ~16.9 GiB memory savings during warmup on TP=1, enabling MTP startup.

### Fix 2: EP global→local expert ID remapping (commit a982d1b)
- **Problem:** MoRI `_prepare()` returns global expert IDs in `dispatch_ids`, but the swiglustep unfused path indexed local weight tensors directly. Under EP=4, local tensors have 72 experts per rank — any global ID ≥ 72 caused IndexError.
- **Solution:** Remap `dispatch_ids` through `self.experts.expert_map` (global→local mapping tensor) before passing to `_swiglustep_unfused_compute()`.
- **Impact:** EP=4 with TP=4 no longer crashes during swiglustep forward.

### Fix 3: MTP layer_types bounds check (commit 53bd575)
- **Problem:** `get_layer_attention_config()` and `get_layer_rope_scaling()` indexed `self.layer_types` with MTP layer indices (45-47), causing IndexError when the list only covered 45 main layers (fallback/default case).
- **Solution:** Bounds check with fallback to `"sliding_attention"` for out-of-range indices.

### Current status
- Code changes committed on branch `add-step3p5-flash`
- All 402 unit tests pass (excluding 29 pre-existing failures in unrelated tests)
- All 61 step3p5-specific tests pass
- **Not yet tested on GPU** — needs MTP startup verification on a node with the model checkpoint

### Verification plan
```bash
python tools/verify_step3p5_mtp.py \
    --model /data/jamesmit/models/Step-3.5-Flash-FP8 \
    --tp 8 --compare
```

## Relationships
- depends-on: [[step3p5-mtp]]
- depends-on: [[step3p5-mtp-debug-2026-05-12]]
- related: [[step3p5-current-status]]
- related: [[step3p5-gaps]]
- related: [[step3p5-inference-benchmarks]]
