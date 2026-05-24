"""Toggle: replace GemmaRMSNorm to match SGLang's bf16 precomputed gemma_weight
behavior, then re-run ATOM and report top-5 logits.

Hypothesis: Step-3.5-Flash was trained against an implementation where
`(1 + weight)` is precomputed in bfloat16 (losing precision when |weight| < 2^-8).
ATOM uses fp32 `(1.0 + weight.float())` per call — more precise, but possibly
not what the model expects.

If patching ATOM to match SGLang flips the argmax to the correct token (565
== ' -' for prompt "What is 2+2?"), then this confirms the precision-trick is
required.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

# Don't enable per-layer capture for this run — we just want the final argmax.
os.environ["ATOM_DIAG_STEP3P5"] = "0"

import torch  # noqa: E402

# Monkey-patch BEFORE the model is loaded.
from atom.model_ops import layernorm as _ln  # noqa: E402

_orig_forward_static = _ln.GemmaRMSNorm.forward_static


def _sglang_style_static(weight, variance_epsilon, x, residual):
    """Match SGLang HIP path: gemma_weight = (weight.data + 1.0) in bf16,
    then x_norm * gemma_weight in bf16. Variance is still computed in fp32.
    """
    orig_dtype = x.dtype
    if residual is not None:
        if orig_dtype == torch.float16:
            x = x.float() + residual.float()
        else:
            x = x + residual
        residual = x

    # Variance computation in fp32 (same as both engines).
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + variance_epsilon)
    x_norm = x_norm.to(orig_dtype)

    # SGLang's trick: precompute gemma_weight = (weight + 1.0) in the weight's
    # dtype (bf16), then multiply in bf16.
    gemma_weight = weight + 1.0  # In bf16, since weight is bf16
    out = x_norm * gemma_weight
    return out if residual is None else (out, residual)


_ln.GemmaRMSNorm.forward_static = staticmethod(_sglang_style_static)
print("[hack] Patched GemmaRMSNorm.forward_static to SGLang-style bf16 gemma_weight")
sys.stdout.flush()

# Now run ATOM
from atom import SamplingParams  # noqa: E402
from atom.model_engine.arg_utils import EngineArgs  # noqa: E402

PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
    "Translate to French: Hello, world.",
]
MODEL = "/data/jamesmit/models/Step-3.5-Flash-FP8"


def main():
    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=1,
        enforce_eager=True,
        kv_cache_dtype="fp8",
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        cudagraph_capture_sizes="[1]",
        level=0,
    )
    llm = engine_args.create_engine()
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=30))
    llm.close()
    results = []
    for prompt, out in zip(PROMPTS, outputs):
        results.append({"prompt": prompt, "completion": out["text"]})
    print("__SUMMARY_BEGIN__")
    print(json.dumps({"engine": "atom-sglang-style-rmsnorm", "results": results}, indent=2))
    print("__SUMMARY_END__")


if __name__ == "__main__":
    main()
