---
name: step3p5-current-status
type: decision
tags: [step3p5, status, accuracy, source-of-truth]
created: 2026-05-06
updated: 2026-05-22
status: active
---

# Step-3.5-Flash Current Status

## Summary
Step-3.5-Flash basic inference is **accuracy-validated**. MTP and EP blockers have been **fixed in code** (2026-05-22): deferred swiglustep bf16 materialisation unblocks MTP startup, and global→local expert ID remapping fixes EP swiglustep. Awaiting GPU validation.

## Details

### Current truth (2026-05-12)
- **Basic inference is correct.** 100-prompt sweep shows 0/100 Chinese corruption, 89/100 first-word match vs vLLM.
- **ATOM is closer to vLLM than SGLang is** (89/100 vs 85/100 first-word match).
- Throughput benchmarks from earlier sessions are now meaningful since accuracy is validated.
- Two remaining features for production readiness: MTP-3 and EP (see below).

### Completed work
- `atom/model_config/step3p5.py`, `atom/models/step3p5.py`, registration in model_runner.py
- Per-layer RoPE theta, partial rotary factor, Q head count — all working
- SwiGLU-step activation fix for MoE layers 43-44 (unfused fallback with custom op)
- Shared expert MLP swiglustep handling
- Weight-loading, tokenization, embedding, config parsing — all validated
- 100-prompt correctness comparison tooling (ATOM vs vLLM vs SGLang)

### Remaining work (priority order)
1. **MTP speculative decoding** — Code fixes applied (commit 53bd575, a982d1b). Awaiting GPU validation. See [[step3p5-mtp-ep-fix-2026-05-22]].
2. **Expert Parallelism (EP)** — EP swiglustep global→local ID fix applied. TP=4 EP should now work. Awaiting GPU validation.

### Accuracy comparison (100 prompts, temperature=0, max_tokens=256)

| Comparison | Exact match | First-word match |
|---|---|---|
| ATOM vs vLLM | 5/100 | 89/100 |
| ATOM vs SGLang | 0/100 | 84/100 |
| SGLang vs vLLM | 0/100 | 85/100 |

| Engine | Chinese code-switching |
|---|---|
| ATOM | 1/100 (model behavior) |
| vLLM | 1/100 (model behavior) |
| SGLang | 16/100 (swiglustep bug) |

### Claims now validated
- Step-3.5-Flash basic inference works end-to-end with correct output quality.
- TP=1 throughput benchmarks are valid (91.7 tok/s decode at C=1).

## Relationships
- depends-on: [[step3p5-swiglustep-fix]]
- depends-on: [[step3p5-flash]]
- related: [[step3p5-engine-comparison]]
- related: [[step3p5-mtp]]
- related: [[step3p5-gaps]]
- related: [[step3p5-logits-comparison]]
