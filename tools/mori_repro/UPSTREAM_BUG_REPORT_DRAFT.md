# MoRI IntraNode dispatch deadlocks at hidden_dim >= 4096 on 8x MI325X

**File at:** https://github.com/ROCm/mori/issues

## Summary

`mori.ops.EpDispatchCombineOp.dispatch()` deadlocks (all ranks 100% GPU, never returns) when `hidden_dim >= ~4096` on 8x MI325X (gfx942), even with no other collective active. The same call passes at `hidden_dim <= 2048`. This is independent of:

- The kernel type (both `IntraNode` and `AsyncLL` are affected).
- The `warp_num_per_block` / `block_num` tuning (tested {16/80}, {16/128}, {4/64}).
- Whether NCCL/RCCL is also in use (passing `NCCL_P2P_DISABLE=1` does not help; pure-MoRI `cycle` mode without any allreduce still deadlocks).
- Whether the call is inside a CUDA graph (we hit it during eager warmup, before any graph capture).

## Environment

- Hardware: 8x AMD MI325X (gfx942), single-node intra-node (xGMI)
- ROCm: 7.2.0
- Python: 3.12
- PyTorch: 2.x (TheRock build)
- MoRI: 1.1.1 (pip `amd_mori==1.1.1`, latest release as of 2026-05-26)
- Required env vars on this node (without them shmem init segfaults):
  - `MORI_RDMA_DEVICES="^bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3"`
  - `MORI_DISABLE_TOPO=1`
- NICs: 4x `bnxt_re*` (Broadcom RoCEv2) — explicitly excluded from MoRI's RDMA list because MoRI's topology probe fails on them.

## Minimal repro

`repro_minimal.py` in this directory. Run with:

```bash
# PASSES (~7s):
MORI_RDMA_DEVICES="^bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3" \
MORI_DISABLE_TOPO=1 \
torchrun --nproc_per_node=8 --master_port=29501 repro_minimal.py \
  --mode cycle --iters 5 --warp-per-block 4 --tokens-per-rank 1024 --hidden 2048

# HANGS forever (passes barrier, never reaches "DONE"):
MORI_RDMA_DEVICES="^bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3" \
MORI_DISABLE_TOPO=1 \
torchrun --nproc_per_node=8 --master_port=29502 repro_minimal.py \
  --mode cycle --iters 5 --warp-per-block 4 --tokens-per-rank 1024 --hidden 4096
```

Mode `cycle` is purely `dispatch` + `combine` per iteration. No NCCL, no compute, no concurrent streams.

## Observed behavior

- All 8 ranks finish `shmem_torch_process_group_init`.
- All 8 ranks finish constructing `EpDispatchCombineOp`.
- All 8 ranks pass `dist.barrier()` (gloo).
- The first call to `dispatch()` enters the kernel, GPU goes to 100%, and never returns.
- `rocm-smi` shows all 8 GPUs at GPU%=100, GPU Memory Read/Write Activity=0%.
- `py-spy dump` (when permissions allow) on each rank shows the host thread blocked in HIP/HSA waiting on a stream/event from the dispatch kernel.

## What we hypothesize (not confirmed in MoRI source)

The dispatch kernel uses `ShmemInt32WaitUntilGreaterThan` (or similar) to wait for peer ranks to write into per-rank shmem mailboxes. At larger `hidden_dim`, the per-token payload region pushes the per-rank signal slot to a different shmem heap fragment than the kernel polls, so the signal is never seen by the waiter and the kernel spins forever.

This is consistent with the threshold being at a specific hidden size (around 2048→4096) and with `MORI_DISABLE_TOPO=1` being required (topology-based heap layout may be wrong on this node, but with topo disabled MoRI may be using a fallback layout that has the boundary issue).

## What we tried that did NOT help

- `NCCL_P2P_DISABLE=1` (no NCCL involved in the failing repro anyway)
- `NCCL_PROTO=Simple NCCL_ALGO=Ring`
- `GPU_MAX_HW_QUEUES=8` (per ROCm/ATOM #506)
- `warp_num_per_block=4` (per a prior MoRI PR claiming 16 hangs on gfx942)
- Smaller `block_num` (64)
- Reducing `max_num_inp_token_per_rank` (tested 1024–16384; all hang at hidden=4096+)
- Moving MoRI to a dedicated CUDA stream (causes HIP error 709 separately)

## What we want from MoRI maintainers

1. Confirm that this is a known issue or reproduces on your test rig at hidden=4096+.
2. Guidance on whether `MORI_DISABLE_TOPO=1` is supposed to be safe — if it changes heap layout, that may be the root cause.
3. Either a fix or a documented `max_hidden_dim` for IntraNode dispatch on gfx942.

## Related issues (apparent prior art / sibling failures)

- **[ROCm/mori#210](https://github.com/ROCm/mori/issues/210)** — `EpDispatchCombineOp` SIGSEGV/OOM at exactly the same `hidden_dim=7168`, 256 experts, top-8 shape on MI355X with sglang-0.5.9-rocm720 + mori-0227-2. The reporter's `MORI_SHMEM_HEAP_SIZE` workaround does not apply directly to our deadlock symptom, but the shape/regime is identical. **Closest match.**
- **[ROCm/mori#168](https://github.com/ROCm/mori/issues/168)** — MoRI-EP **internode** dispatch hang on MI300X+CX7 with assertion `lanePe < worldSize`. Different transport but same dispatch-kernel-hang failure class.
- **[ROCm/mori#276](https://github.com/ROCm/mori/issues/276)** — Documents `MORI_DISABLE_P2P=ON` as a known workaround for intranode P2P routing issues on rail-optimized fabrics. (Untested here; worth trying.)
- **[ROCm/aiter#346](https://github.com/ROCm/aiter/issues/346)** — `test_moe.py` stuck at `hidden_dim=8192` on MI308X. Same "stuck at specific large hidden_dim" signature for AITER MoE kernels. Open since Apr 2025.
- **[vllm-project/vllm#43547](https://github.com/vllm-project/vllm/issues/43547)** — Different EP backend (`allgather_reducescatter`) but the same structural failure: divergent collective on one rank → TP peers deadlock. Validates that this class of bug is recognized across MoE EP backends.

Our standalone repro at `hidden_dim>=4096` with pure dispatch+combine (no concurrent allreduce, no NCCL, no compute) appears to isolate the bug more cleanly than the existing reports and may be useful to attach to mori#210.
