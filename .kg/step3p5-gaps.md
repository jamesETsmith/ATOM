---
name: step3p5-gaps
type: process
tags: [gaps, new-work, step3p5, engineering]
created: 2026-04-30
updated: 2026-05-22
status: active
---

# Step 3.5 Flash — Engineering Gaps

## Summary
Remaining features for Step-3.5-Flash production readiness. Basic inference gaps are all resolved. Two major features remain: MTP-3 speculative decoding and expert parallelism.

## Details

### ~~Per-Layer RoPE Theta~~ (DONE)
Implemented. Per-layer rope_theta list indexed in Step3p5Attention.__init__.

### ~~Per-Layer Partial Rotary Factor~~ (DONE)
Implemented. Per-layer partial_rotary_factors indexed in attention constructor.

### ~~Per-Layer Q Head Count~~ (DONE)
Implemented. Full-attention layers use 64 Q heads, sliding layers use 96.

### ~~SwiGLU Clamping / SwiGLU-Step~~ (DONE — commit 6313028)
Fixed with unfused MoE fallback for layers 43-44. See [[step3p5-swiglustep-fix]] for details.
- Routed experts: unfused path dequantizes FP8 to bf16, applies `silu(gate).clamp(max=L) * up.clamp(-L, L)`
- Shared expert: swiglustep applied inside MLP activation
- Registered as custom op with BMM path (CUDA graph) and expert-loop path (eager)

### MTP-3 (P0, Medium Effort — CODE FIXES APPLIED)
3 MTP layers (45-47) for speculative decoding of 3 extra tokens per step. Expected ~2-3x throughput improvement. See [[step3p5-mtp]] for architecture details.

**Status:** Deferred bf16 materialisation fix (commit 53bd575) unblocks MTP startup OOM. Awaiting GPU validation. See [[step3p5-mtp-ep-fix-2026-05-22]].

### Expert Parallelism (P1, Medium-High Effort — CODE FIX APPLIED)
Distribute 288 experts across multiple GPUs. The benchmark data shows EP configurations (TP=4 EP, TP=8 EP) but TP=4 was crashing due to swiglustep global→local expert ID mismatch.

**Status:** EP global→local ID remapping fix applied (commit a982d1b). Awaiting GPU validation at TP=4.

## Relationships
- depends-on: [[step3p5-flash]]
- used-by: [[step3p5-implementation-plan]]
- related: [[atom-attention-support]]
- related: [[atom-moe-support]]
- related: [[step3p5-swiglustep-fix]]
- related: [[step3p5-mtp]]
