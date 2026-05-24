"""Verify fix: ATOM with SGLang-style bf16-precomputed gemma_weight EVERYWHERE.

Patches:
1. GemmaRMSNorm.forward_static — bf16 (weight + 1.0) precomputed each call.
2. Triton kernel _fused_qkv_norm_rope_cache_kernel — wrap caller to pre-add 1.0
   to weight in bf16 before passing into the kernel, then disable the kernel's
   internal `(1.0 + qw)` step by patching its source (or, simpler, by replacing
   the wrapper function entirely).

Approach for (2): replace `triton_fused_norm_rope_cache` with a Python
implementation that calls a SEPARATE GemmaRMSNorm + rotary_emb + AITER
reshape_and_cache. Slower but guaranteed correct semantics.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
os.environ["ATOM_DIAG_STEP3P5"] = "0"

import torch  # noqa: E402

# ----- Patch 1: GemmaRMSNorm.forward_static -----
from atom.model_ops import layernorm as _ln  # noqa: E402


def _sglang_style_static(weight, variance_epsilon, x, residual):
    """Match SGLang HIP path: gemma_weight = (weight + 1.0) precomputed in
    bfloat16 dtype (since weight is bf16), then x_norm * gemma_weight in bf16.
    Variance computed in fp32.
    """
    orig_dtype = x.dtype
    if residual is not None:
        if orig_dtype == torch.float16:
            x = x.float() + residual.float()
        else:
            x = x + residual
        residual = x

    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + variance_epsilon)
    x_norm_bf16 = x_norm.to(orig_dtype)

    # bf16 (weight + 1.0) precomputed; multiply in bf16
    gemma_weight = weight + 1.0  # bf16 + fp scalar = bf16
    out = x_norm_bf16 * gemma_weight
    return out if residual is None else (out, residual)


_ln.GemmaRMSNorm.forward_static = staticmethod(_sglang_style_static)
print("[hack] Patched GemmaRMSNorm.forward_static (bf16 gemma_weight)")
sys.stdout.flush()

# ----- Patch 2: Triton fused norm+RoPE+cache kernel wrapper -----
# Replace it with a Python implementation: q_norm(q) -> rotary_emb(q,k) ->
# aiter.reshape_and_cache. This forces the Q/K norms to use our patched
# GemmaRMSNorm which now uses bf16 gemma_weight.

from atom.model_ops import triton_fused_qkv_norm_rope_cache as _tfk  # noqa: E402
import aiter  # noqa: E402


def _python_norm_rope_cache(
    q, k, v, positions, q_norm, k_norm, rotary_emb,
    num_heads, num_kv_heads, head_dim,
    k_cache, v_cache, k_scale, v_scale,
    slot_mapping, kv_cache_dtype,
):
    # Reshape q [T, num_heads*head_dim] -> [T*num_heads, head_dim] for per-head norm
    T = q.shape[0]
    q_for_norm = q.reshape(-1, head_dim)
    k_for_norm = k.reshape(-1, head_dim)
    # Apply GemmaRMSNorm (now patched to bf16 gemma_weight semantics).
    q_normed = q_norm(q_for_norm).reshape(T, num_heads * head_dim)
    k_normed = k_norm(k_for_norm).reshape(T, num_kv_heads * head_dim)
    # RoPE
    q_roped, k_roped = rotary_emb(positions, q_normed, k_normed)
    # KV cache write (FP8 quant)
    if kv_cache_dtype == "fp8":
        # The Triton kernel writes a SHUFFLE-layout v_cache passed in as
        # [B, H, BS//X, HD, X]. AITER's reshape_and_cache_with_pertoken_quant
        # expects asm_layout=True and v_cache as [B, H, HD, BS]. We restore
        # the original layout view that the caller had before shuffling.
        # v_cache from caller is [B, H, BS//X, HD, X] -> view back to [B, H, HD, BS]
        x = 16 // v_cache.element_size()
        n, nh, bs_div_x, hd, x_size = v_cache.shape
        bs = bs_div_x * x_size
        v_cache_orig = v_cache.view(n, nh, hd, bs)
        # k expected as [T, num_kv_heads, head_dim]
        k_for_cache = k_roped.view(T, num_kv_heads, head_dim)
        v_for_cache = v.view(T, num_kv_heads, head_dim)
        aiter.reshape_and_cache_with_pertoken_quant(
            k_for_cache, v_for_cache, k_cache, v_cache_orig, k_scale, v_scale,
            slot_mapping, asm_layout=True,
        )
    else:
        k_for_cache = k_roped.view(T, num_kv_heads, head_dim)
        v_for_cache = v.view(T, num_kv_heads, head_dim)
        aiter.reshape_and_cache(
            k_for_cache, v_for_cache, k_cache, v_cache,
            slot_mapping, kv_cache_dtype="auto",
            k_scale=None, v_scale=None, asm_layout=True,
        )
    return q_roped, k_roped


_tfk.triton_fused_norm_rope_cache = _python_norm_rope_cache
# Also re-patch in attention_mha if it's already been imported/captured.
import atom.model_ops.attention_mha as _amha  # noqa: E402
# The import is `from atom.model_ops.triton_fused_qkv_norm_rope_cache import triton_fused_norm_rope_cache`
# done LAZILY inside the function (line 142-144). So our module-level patch is enough.
print("[hack] Patched triton_fused_norm_rope_cache -> python fallback")
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
    print(json.dumps({"engine": "atom-bf16-gemma-everywhere", "results": results}, indent=2))
    print("__SUMMARY_END__")


if __name__ == "__main__":
    main()
