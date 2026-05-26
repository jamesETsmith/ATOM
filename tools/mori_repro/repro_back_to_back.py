"""Standalone MoRI back-to-back dispatch+combine repro.

Goal: determine whether N consecutive dispatch/combine cycles on a single
shared EpDispatchCombineOp instance deadlock when (a) a sync is inserted
between them, or (b) the cycles are queued and synced at the end.

Run with:
  MORI_RDMA_DEVICES="^bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3" \
  MORI_DISABLE_TOPO=1 \
  torchrun --nproc_per_node=8 tools/mori_repro/repro_back_to_back.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def log(rank: int, msg: str) -> None:
    sys.stderr.write(f"[r{rank} {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=45,
                        help="Number of back-to-back dispatch/combine cycles")
    parser.add_argument("--tokens-per-rank", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sync-every", type=int, default=0,
                        help="If >0, torch.cuda.synchronize() after this many iters")
    parser.add_argument("--call-reset", action="store_true",
                        help="Pass call_reset=True to combine()")
    parser.add_argument("--final-sync-only", action="store_true",
                        help="Only sync at the end (mimics layers 3-42 pattern)")
    parser.add_argument("--swiglu-sync-at", type=int, default=-1,
                        help="Iter index at which to do an .item() sync mid-loop (mimics layer 43)")
    args = parser.parse_args()

    # ---- Distributed init (gloo for CPU shmem bootstrap) -----------------
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # MoRI requires "mori" process group registration + shmem init
    import mori
    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    log(rank, f"shmem init done, world={world_size}")

    # ---- Construct MoRI op matching Step-3.5-Flash config ----------------
    local_experts = args.num_experts // world_size
    mori_config = mori.ops.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        data_type=torch.bfloat16,
        hidden_dim=args.hidden,
        scale_dim=0,
        scale_type_size=torch.float32.itemsize,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=args.tokens_per_rank,
        num_experts_per_rank=local_experts,
        num_experts_per_token=args.topk,
        warp_num_per_block=16,
        block_num=80,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        rdma_block_num=0,
        gpu_per_node=8,
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)
    log(rank, f"mori_op created: local_experts={local_experts}, "
              f"topk={args.topk}, tokens={args.tokens_per_rank}")

    # ---- Build random input ----------------------------------------------
    M = args.tokens_per_rank
    H = args.hidden
    torch.manual_seed(42 + rank)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

    # Random topk routing
    topk_ids = torch.randint(0, args.num_experts, (M, args.topk),
                             device="cuda", dtype=torch.int32)
    topk_weights = torch.rand(M, args.topk, device="cuda",
                              dtype=torch.float32)

    # Dummy empty scales (fp16 path → scales unused)
    scales = torch.empty(0, device="cuda", dtype=torch.float32)

    dist.barrier()
    log(rank, "barrier passed, starting back-to-back cycles")

    # ---- Run loop --------------------------------------------------------
    t0 = time.time()
    for i in range(args.iters):
        # DISPATCH
        try:
            disp_out = mori_op.dispatch(x, topk_weights, scales, topk_ids)
            disp_x, disp_w, disp_s, disp_ids, disp_recv_count = disp_out
        except Exception as e:
            log(rank, f"iter {i} dispatch FAILED: {e}")
            return 1

        # Pretend to compute (zeros same shape as disp_x to avoid wasting time)
        fake_out = torch.zeros_like(disp_x)

        # Mimic swiglustep layer 43 mid-loop sync
        if i == args.swiglu_sync_at:
            log(rank, f"iter {i}: swiglustep-style .any() sync ENTER")
            mask = (disp_ids == 0).any() if disp_ids.numel() > 0 else torch.tensor(False, device="cuda")
            r = mask.item()
            log(rank, f"iter {i}: swiglustep-style .any() sync DONE, result={r}")

        # COMBINE (back to caller)
        try:
            combine_out = mori_op.combine(
                fake_out,
                None,
                disp_ids,
                call_reset=args.call_reset,
            )
        except Exception as e:
            log(rank, f"iter {i} combine FAILED: {e}")
            return 2

        # Intermediate sync
        if args.sync_every > 0 and (i + 1) % args.sync_every == 0:
            log(rank, f"iter {i}: intermediate sync ENTER")
            torch.cuda.synchronize()
            log(rank, f"iter {i}: intermediate sync DONE")

        if (i + 1) % 5 == 0:
            log(rank, f"iter {i+1}/{args.iters} (no sync) queued")

    # Final sync
    log(rank, "final sync ENTER")
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    log(rank, f"final sync DONE, elapsed={elapsed:.2f}s")

    dist.barrier()
    log(rank, "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
