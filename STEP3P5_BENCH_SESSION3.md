# Step-3.5-Flash-FP8 TP=8 Benchmark Comparison — Session 3 (2026-05-26)

8x MI325X (gfx942), ROCm 7.2.0, identical workload (random 1024 in / 1024 out, 256 prompts, max-concurrency 64, range ratio 0.8).

| Metric | TP=8 baseline | TP=8 + `NCCL_P2P_DISABLE=1` | Delta |
|---|---|---|---|
| Output throughput | **2069 tok/s** | 1976 tok/s | -4.5% |
| Total throughput | **4149 tok/s** | 3963 tok/s | -4.5% |
| Mean TPOT | 27.95 ms | 28.29 ms | +1.2% |
| Median TPOT | 28.95 ms | 29.26 ms | +1.1% |
| Mean TTFT | 1751 ms | 2763 ms | +58% |
| Median TTFT | 338 ms | 332 ms | -1.8% |
| P99 TTFT | 7614 ms | 12588 ms | +65% |
| Mean E2EL | 27428 ms | 28748 ms | +4.8% |
| P99 E2EL | 36895 ms | 41742 ms | +13% |

## Takeaways

1. **TP=8 baseline is the recommended production config** for Step-3.5-Flash-FP8 on 8x MI325X today.
2. `NCCL_P2P_DISABLE=1` (the partial fix for the EP MoRI deadlock at small `hidden_dim`) costs ~4.5% steady-state throughput and ~65% worse P99 TTFT under concurrent load. **Do not enable on TP-only deployments.**
3. EP (`--enable-expert-parallel`) remains blocked by the MoRI IntraNode dispatch deadlock at `hidden_dim >= 4096` (see `tools/mori_repro/UPSTREAM_BUG_REPORT_DRAFT.md`).

## Files

- `bench_step3p5_tp_baseline_session3.json` — TP=8 baseline, no env overrides
- `bench_step3p5_tp_p2pdisable_session3.json` — TP=8 + `NCCL_P2P_DISABLE=1`

Both used identical benchmark config: dataset=random, isl=1024, osl=1024, range_ratio=0.8, num_prompts=256, max_concurrency=64, request_rate=inf, ignore_eos.
