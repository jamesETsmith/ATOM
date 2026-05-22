---
name: step3p5-gpu-validation-2026-05-22
type: process
tags: [step3p5, validation, gpu, mtp, ep, mori]
created: 2026-05-22
updated: 2026-05-22
status: active
---

# Step-3.5-Flash GPU Validation Results (2026-05-22)

## Summary
Validated Step-3.5-Flash-FP8 on 8x MI325X. Baseline TP=8 and MTP TP=8 both PASS (5/5 first-word match). EP TP=8 gets past our code fixes but crashes with SIGSEGV inside MoRI's native shmem initialization. The two ATOM-level EP bugs are fixed; the remaining blocker is MoRI's native C++ layer segfaulting during shared-memory setup on this node.

## Details

### Test Results

| Test | Status | Duration | Detail |
|------|--------|----------|--------|
| Baseline TP=8 | **PASS** | 180s | 5/5 prompts OK |
| MTP TP=8 | **PASS** | 200s | First-word match: 5/5 |
| EP TP=8 | **FAIL** | 600s | SIGSEGV in MoRI shmem init |

### MTP First-Word Comparison (5/5 match)

| Prompt | Baseline | MTP | Match |
|--------|----------|-----|-------|
| What is the capital of France? | \|Paris | \|Paris | YES |
| Explain quantum computing... | - | - | YES |
| Write a Python function... | The | The | YES |
| The meaning of life is | a | a | YES |
| List 5 benefits of exercise | 1. | 1. | YES |

### EP Failure Analysis

**Progress made**: Our `use_all2all_kernels` fix works — MoRI is now being initialized. The code reaches `init_prepare_finalize()` → `_maybe_make_prepare_finalize()` → MoRI buffer registration. Weight loading completes for all 44 shards. Swiglustep layers correctly report `ep=True`.

**Crash point**: After weight loading, during MoRI's `shmem_torch_process_group_init()`, one ModelRunner process (randomly rank 6 or 7) dies with `exitcode=-11` (SIGSEGV). This is inside MoRI's native C++ shared-memory layer, not in ATOM Python code.

**Root cause hypothesis**: The `amd-mori` pip package (v1.1.1) JIT-compiles `.hip` kernels at first use. The shmem kernel compilation requires `infiniband/verbs.h` (RDMA headers) which were missing — we manually extracted them from the `libibverbs-dev` deb package. The RDMA transport itself is not needed (single-node, `gpu_per_node=8`), but MoRI still compiles the RDMA transport module. The segfault may be due to missing RDMA runtime support or an incompatibility between the pip-installed MoRI and this node's ROCm/driver version.

**MoRI installation**: `pip install amd-mori` (v1.1.1) — on PyPI. Required `CPLUS_INCLUDE_PATH` set to point at manually-provided `infiniband/verbs.h` headers.

### Environment

- Node: quanta-ccs-aus-f01-40.adc.amd.com
- GPUs: 8x MI325X (gfx942), 256 GiB VRAM each
- Model: Step-3.5-Flash-FP8 (196B params, 288 experts, top-8)
- ATOM branch: add-step3p5-flash (commit 4369848)
- amd-mori: 1.1.1 (pip)
- ROCm: 7.2.0

### What's Fixed (ATOM code)

1. `use_all2all_kernels` gate (`moe.py:82`): removed `dp_size > 1` — allows MoRI for pure EP
2. Token truncation (`modular_kernel.py:336`, `step3p5.py:311`): `num_dispatchers` instead of `dp_size`
3. Both fixes committed in 7ba0a3a, validation script in 4369848

### What Remains (MoRI infrastructure)

- MoRI segfaults during shmem init on this node
- Likely needs a node with proper RDMA/InfiniBand dev stack or a MoRI version that handles missing RDMA gracefully
- The ATOM recipe (`recipes/Step-3.5-Flash.md`) claims EP works with `-tp 8 --enable-expert-parallel` — presumably tested on a properly equipped multi-node cluster

## Relationships
- depends-on: [[step3p5-ep-dispatch-bug]]
- related: [[step3p5-flash]]
