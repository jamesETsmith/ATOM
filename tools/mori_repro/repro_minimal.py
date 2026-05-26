"""Minimal MoRI hang repro. Strips compute and isolates dispatch+allreduce.

Hypothesis: a MoRI IntraNode dispatch concurrent with an RCCL/NCCL allreduce
deadlocks because both consume the same GPU sync primitives.

Modes (one at a time, controlled by --mode):
  dispatch_only         : just dispatch, no allreduce. Should pass.
  allreduce_only        : just allreduce, no dispatch. Should pass.
  dispatch_then_allreduce: dispatch, sync, allreduce. Sequential.
  dispatch_and_allreduce: dispatch + allreduce on same default stream (no sync between).
  cycle                 : N iterations of dispatch + combine, no allreduce.
  cycle_with_allreduce  : N iterations of dispatch + allreduce + combine.

Use --iters to control iteration count.
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
    parser.add_argument(
        "--mode",
        choices=[
            "dispatch_only",
            "allreduce_only",
            "dispatch_then_allreduce",
            "dispatch_and_allreduce",
            "cycle",
            "cycle_with_allreduce",
            "cycle_with_allreduce_dedicated_stream",
            "cycle_with_allreduce_record_event",
            "asyncll_cycle_with_allreduce",
        ],
        default="cycle_with_allreduce",
    )
    parser.add_argument("--iters", type=int, default=45)
    parser.add_argument("--tokens-per-rank", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--warp-per-block",
        type=int,
        default=16,
        help="MoRI warp_num_per_block. PR #286 says 16 hangs on gfx942; use 4.",
    )
    parser.add_argument("--block-num", type=int, default=80)
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    need_nccl = args.mode in (
        "allreduce_only",
        "dispatch_then_allreduce",
        "dispatch_and_allreduce",
        "cycle_with_allreduce",
        "cycle_with_allreduce_dedicated_stream",
        "cycle_with_allreduce_record_event",
        "asyncll_cycle_with_allreduce",
    )
    nccl_group = None
    if need_nccl:
        nccl_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
        log(rank, "nccl group created")

    import mori

    torch._C._distributed_c10d._register_process_group("mori", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori")
    log(rank, f"shmem init done, world={world_size}, mode={args.mode}")

    local_experts = args.num_experts // world_size
    kernel_type = (
        mori.ops.EpDispatchCombineKernelType.AsyncLL
        if args.mode.startswith("asyncll_")
        else mori.ops.EpDispatchCombineKernelType.IntraNode
    )
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
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        kernel_type=kernel_type,
        rdma_block_num=0,
        gpu_per_node=8,
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)
    log(rank, f"mori_op created: local_experts={local_experts}")

    M = args.tokens_per_rank
    H = args.hidden
    torch.manual_seed(42 + rank)

    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(
        0, args.num_experts, (M, args.topk), device="cuda", dtype=torch.int32
    )
    topk_weights = torch.rand(M, args.topk, device="cuda", dtype=torch.float32)
    scales = torch.empty(0, device="cuda", dtype=torch.float32)
    ar_buf = torch.ones(1024, device="cuda", dtype=torch.float32)

    dist.barrier()
    log(rank, f"barrier passed, starting mode={args.mode}")
    t0 = time.time()

    if args.mode == "dispatch_only":
        for i in range(args.iters):
            mori_op.dispatch(x, topk_weights, scales, topk_ids)
        torch.cuda.synchronize()
    elif args.mode == "allreduce_only":
        for i in range(args.iters):
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
        torch.cuda.synchronize()
    elif args.mode == "dispatch_then_allreduce":
        for i in range(args.iters):
            mori_op.dispatch(x, topk_weights, scales, topk_ids)
            torch.cuda.synchronize()
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
            torch.cuda.synchronize()
    elif args.mode == "dispatch_and_allreduce":
        for i in range(args.iters):
            mori_op.dispatch(x, topk_weights, scales, topk_ids)
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
        torch.cuda.synchronize()
    elif args.mode == "cycle":
        for i in range(args.iters):
            dout = mori_op.dispatch(x, topk_weights, scales, topk_ids)
            disp_x, disp_w, disp_s, disp_ids, _ = dout
            mori_op.combine(torch.zeros_like(disp_x), None, disp_ids)
            if (i + 1) % 10 == 0:
                log(rank, f"iter {i+1}/{args.iters} queued")
        torch.cuda.synchronize()
    elif args.mode == "cycle_with_allreduce":
        for i in range(args.iters):
            dout = mori_op.dispatch(x, topk_weights, scales, topk_ids)
            disp_x, disp_w, disp_s, disp_ids, _ = dout
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
            mori_op.combine(torch.zeros_like(disp_x), None, disp_ids)
            if (i + 1) % 10 == 0:
                log(rank, f"iter {i+1}/{args.iters} queued")
        torch.cuda.synchronize()
    elif args.mode == "cycle_with_allreduce_dedicated_stream":
        # MoRI on its own stream, allreduce on default stream.
        # Use stream.wait_stream() to make MoRI consumers wait on the comm stream.
        mori_stream = torch.cuda.Stream()
        # WARMUP: do one dispatch/combine on default stream to let MoRI initialize
        # internal state (handle compilation, signal buffer setup).
        warm_d = mori_op.dispatch(x, topk_weights, scales, topk_ids)
        mori_op.combine(torch.zeros_like(warm_d[0]), None, warm_d[3])
        torch.cuda.synchronize()
        log(rank, "warmup on default stream done")
        for i in range(args.iters):
            # Make MoRI stream wait for any pending work on default stream (inputs ready)
            mori_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(mori_stream):
                dout = mori_op.dispatch(x, topk_weights, scales, topk_ids)
                disp_x, disp_w, disp_s, disp_ids, _ = dout
            # Default stream waits for dispatch before any subsequent ops on disp_x
            torch.cuda.current_stream().wait_stream(mori_stream)
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
            mori_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(mori_stream):
                mori_op.combine(torch.zeros_like(disp_x), None, disp_ids)
            torch.cuda.current_stream().wait_stream(mori_stream)
            if (i + 1) % 10 == 0:
                log(rank, f"iter {i+1}/{args.iters} queued (dedicated stream)")
        torch.cuda.synchronize()
    elif args.mode == "cycle_with_allreduce_record_event":
        # Simulate the dispatch-then-allreduce-then-combine pattern
        # but use record/wait events instead of full synchronize.
        for i in range(args.iters):
            dout = mori_op.dispatch(x, topk_weights, scales, topk_ids)
            disp_x, disp_w, disp_s, disp_ids, _ = dout
            ev = torch.cuda.Event()
            ev.record()
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
            ev.wait()  # wait for dispatch before combine
            mori_op.combine(torch.zeros_like(disp_x), None, disp_ids)
            if (i + 1) % 10 == 0:
                log(rank, f"iter {i+1}/{args.iters} queued (event sync)")
        torch.cuda.synchronize()
    elif args.mode == "asyncll_cycle_with_allreduce":
        # AsyncLL uses split send/recv kernels (CU-free) instead of single dispatch.
        for i in range(args.iters):
            dout = mori_op.dispatch_send(x, topk_weights, scales, topk_ids)
            disp_x, disp_w, disp_s, disp_ids, _ = dout
            dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM, group=nccl_group)
            mori_op.dispatch_recv()
            mori_op.combine_send(torch.zeros_like(disp_x), None, disp_ids)
            mori_op.combine_recv()
            if (i + 1) % 10 == 0:
                log(rank, f"iter {i+1}/{args.iters} queued (asyncll)")
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    log(rank, f"DONE elapsed={elapsed:.2f}s")
    dist.barrier()
    log(rank, "PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
