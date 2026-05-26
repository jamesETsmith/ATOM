"""Realistic MoRI repro that interleaves AITER moe_sorting + GEMM between
dispatch/combine cycles, mimicking the actual model forward pass.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.distributed as dist


def log(rank: int, msg: str) -> None:
    sys.stderr.write(f"[r{rank} {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=45)
    parser.add_argument("--tokens-per-rank", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--inter", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--swiglu-sync-at", type=int, default=42)
    parser.add_argument("--allreduce-each-iter", action="store_true")
    parser.add_argument("--moe-sort-each-iter", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # also need nccl for allreduce
    if args.allreduce_each_iter:
        # Create a separate NCCL/RCCL group for collectives
        nccl_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    else:
        nccl_group = None

    import mori

    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    log(rank, f"shmem init done, world={world_size}")

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
    log(rank, f"mori_op created: local_experts={local_experts}")

    M = args.tokens_per_rank
    H = args.hidden
    INTER = args.inter
    torch.manual_seed(42 + rank)

    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0, args.num_experts, (M, args.topk), device="cuda", dtype=torch.int32
    )
    topk_weights = torch.rand(M, args.topk, device="cuda", dtype=torch.float32)
    scales = torch.empty(0, device="cuda", dtype=torch.float32)

    # Fake expert weights
    w13 = torch.randn(local_experts, 2 * INTER, H, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(local_experts, H, INTER, device="cuda", dtype=torch.bfloat16)

    if args.moe_sort_each_iter:
        try:
            import aiter  # noqa: F401
            from aiter import moe_sorting_fwd  # noqa: F401
        except ImportError:
            log(rank, "aiter not available, skipping moe_sorting")
            args.moe_sort_each_iter = False

    dist.barrier()
    log(rank, "barrier passed, starting realistic cycles")

    t0 = time.time()
    for i in range(args.iters):
        # DISPATCH
        disp_out = mori_op.dispatch(x, topk_weights, scales, topk_ids)
        disp_x, disp_w, disp_s, disp_ids, disp_recv_count = disp_out

        # Optional collective alongside MoRI (this is what real model does)
        if args.allreduce_each_iter and nccl_group is not None:
            tmp = torch.ones(1024, device="cuda", dtype=torch.float32)
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=nccl_group)

        # Fake expert compute: bmm × 2 + silu + bmm
        # Use a smaller chunk so it doesn't take forever
        if disp_x.shape[0] > 0:
            local_topk_ids = disp_ids % local_experts  # remap to local
            for eid in range(
                min(local_experts, 4)
            ):  # only first 4 experts to save time
                mask = (
                    (local_topk_ids == eid).any(dim=1)
                    if local_topk_ids.dim() > 1
                    else (local_topk_ids == eid)
                )
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                x_sel = disp_x[idx].to(torch.bfloat16)
                gate_up = x_sel @ w13[eid].t()
                gate = gate_up[:, :INTER]
                up = gate_up[:, INTER : 2 * INTER]
                act = torch.nn.functional.silu(gate) * up
                out = act @ w2[eid].t()
                _ = out.sum()  # to prevent dead code elimination
        else:
            log(rank, f"iter {i}: disp_x empty, skipping compute")

        fake_out = torch.zeros_like(disp_x)

        # Mimic layer 43 mid-loop sync
        if i == args.swiglu_sync_at:
            log(rank, f"iter {i}: SYNC ENTER (mimics swiglustep layer 43)")
            torch.cuda.synchronize()
            log(rank, f"iter {i}: SYNC DONE")

        # COMBINE
        mori_op.combine(fake_out, None, disp_ids)

        if (i + 1) % 10 == 0:
            log(rank, f"iter {i+1}/{args.iters} queued")

    log(rank, "final sync ENTER")
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    log(rank, f"final sync DONE, elapsed={elapsed:.2f}s")

    dist.barrier()
    log(rank, "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
