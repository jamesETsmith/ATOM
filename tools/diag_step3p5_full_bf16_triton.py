"""Option B: full bf16 GemmaRMSNorm test with patched Triton kernel.

This patches both:
  1. `GemmaRMSNorm.forward_static` (covers input/post-attention/final norms)
  2. `triton_fused_norm_rope_cache` (covers the q_norm/k_norm path inside attention)

For #2, we install a *new* Triton kernel that is byte-for-byte identical to
the original except `(1 + qw)` and `(1 + kw)` are computed in bf16 first
(matching SGLang's precomputed `gemma_weight = weight + 1.0` buffer) before
being upcast to fp32 for the multiply.

Critically this avoids the AITER JIT path entirely (Triton kernels compile
in-process), so it sidesteps the `module_rope_pos_fwd` build failure that
killed the prior `diag_step3p5_full_bf16_gemma.py` attempt.

If multi-prompt outputs come back clean (no Chinese-character corruption),
the bf16-(1+w) hypothesis is confirmed for both norm paths and a Step-3.5-
specific production fix is justified.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
os.environ["ATOM_DIAG_STEP3P5"] = "0"

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# Counters to verify both monkey-patches actually fire on every relevant call.
_PATCH_COUNTERS = {
    "gemma_rms_static_calls": 0,
    "gemma_rms_static_unique_shapes": set(),
    "triton_kernel_calls": 0,
    "triton_kernel_unique_shapes": set(),
}

# ---------------------------------------------------------------------------
# Patch 1: GemmaRMSNorm.forward_static -> bf16 (weight + 1.0)
# ---------------------------------------------------------------------------
from atom.model_ops import layernorm as _ln  # noqa: E402


def _sglang_style_static(weight, variance_epsilon, x, residual):
    _PATCH_COUNTERS["gemma_rms_static_calls"] += 1
    _PATCH_COUNTERS["gemma_rms_static_unique_shapes"].add(
        (tuple(x.shape), tuple(weight.shape), str(x.dtype), str(weight.dtype),
         residual is not None)
    )
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
    x_norm = x_norm.to(orig_dtype)
    gemma_weight = weight + 1.0  # bf16
    out = x_norm * gemma_weight
    return out if residual is None else (out, residual)


_ln.GemmaRMSNorm.forward_static = staticmethod(_sglang_style_static)


def _patched_forward_cuda(self, x, residual=None):
    # Bypass torch.compile so the counter isn't traced away.
    return self.forward_native(x, residual)


_ln.GemmaRMSNorm.forward_cuda = _patched_forward_cuda
print("[hack] Patched GemmaRMSNorm.forward_static -> bf16 gemma_weight")
print("[hack] Patched GemmaRMSNorm.forward_cuda -> direct native (no torch.compile)")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Patch 2: triton_fused_norm_rope_cache -> bf16 (1+w) Triton kernel
# ---------------------------------------------------------------------------
from atom.model_ops import triton_fused_qkv_norm_rope_cache as _tfqnrc  # noqa: E402


@triton.jit
def _fused_qkv_norm_rope_cache_kernel_bf16(
    q_ptr, q_stride_t,
    k_ptr, k_stride_t,
    v_ptr, v_stride_t,
    q_out_ptr, k_out_ptr,
    qw_ptr, kw_ptr,
    cos_cache_ptr, sin_cache_ptr,
    cos_sin_stride_pos,
    pos_ptr,
    k_cache_ptr, v_cache_ptr,
    kc_stride_block, kc_stride_head, kc_stride_dx, kc_stride_slot, kc_stride_x,
    vc_stride_block, vc_stride_head, vc_stride_sc, vc_stride_d, vc_stride_x,
    k_scale_ptr, v_scale_ptr,
    ks_stride_block, ks_stride_head,
    vs_stride_block, vs_stride_head,
    slot_mapping_ptr,
    pos_stride_row,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_DIM_HALF: tl.constexpr,
    IS_FP8: tl.constexpr,
    MROPE_S0: tl.constexpr = 0,
    MROPE_S1: tl.constexpr = 0,
    IS_MROPE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    total_heads = num_heads + num_kv_heads
    token_id = pid // total_heads
    head_id = pid % total_heads

    d_offs = tl.arange(0, BLOCK_D)

    if head_id < num_heads:
        h = head_id
        q_in_offset = token_id * q_stride_t + h * BLOCK_D

        q = tl.load(q_ptr + q_in_offset + d_offs).to(tl.float32)

        variance = tl.sum(q * q, axis=0) / BLOCK_D
        q_normed = q * tl.math.rsqrt(variance + eps)
        # ===== BF16 (1 + qw) =====
        qw_bf16 = tl.load(qw_ptr + d_offs)         # native bf16
        qw_plus1 = (qw_bf16 + 1.0).to(tl.float32)  # add in bf16, then upcast
        q_normed = q_normed * qw_plus1

        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)),
        )

        if IS_MROPE:
            pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
            pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
            pos_per_dim = tl.where(
                d_cos_idx < MROPE_S0,
                pos_t,
                tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w),
            )
            cos_base = pos_per_dim * cos_sin_stride_pos
        else:
            pos = tl.load(pos_ptr + token_id)
            cos_base = pos * cos_sin_stride_pos

        cos_vals = tl.load(cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0).to(tl.float32)
        sin_vals = tl.load(sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0).to(tl.float32)

        gather_idx = tl.where(
            first_half_mask,
            d_offs + ROTARY_DIM_HALF,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
        )
        q_gathered_raw = tl.load(q_ptr + q_in_offset + gather_idx).to(tl.float32)
        # ===== BF16 (1 + qw_gathered) =====
        qw_g_bf16 = tl.load(qw_ptr + gather_idx)
        qw_g_plus1 = (qw_g_bf16 + 1.0).to(tl.float32)
        q_gathered_normed = q_gathered_raw * tl.math.rsqrt(variance + eps) * qw_g_plus1
        q_rot = tl.where(first_half_mask, -q_gathered_normed, q_gathered_normed)
        q_rot = tl.where(rot_mask, q_rot, 0.0)

        q_roped = q_normed * cos_vals + q_rot * sin_vals

        q_out_offset = token_id * (num_heads * BLOCK_D) + h * BLOCK_D
        tl.store(q_out_ptr + q_out_offset + d_offs, q_roped.to(q_out_ptr.dtype.element_ty))
    else:
        kv_h = head_id - num_heads

        k_in_offset = token_id * k_stride_t + kv_h * BLOCK_D
        k = tl.load(k_ptr + k_in_offset + d_offs).to(tl.float32)

        k_variance = tl.sum(k * k, axis=0) / BLOCK_D
        k_normed = k * tl.math.rsqrt(k_variance + eps)
        # ===== BF16 (1 + kw) =====
        kw_bf16 = tl.load(kw_ptr + d_offs)
        kw_plus1 = (kw_bf16 + 1.0).to(tl.float32)
        k_normed = k_normed * kw_plus1

        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)),
        )

        if IS_MROPE:
            pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
            pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
            pos_per_dim = tl.where(
                d_cos_idx < MROPE_S0,
                pos_t,
                tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w),
            )
            cos_base = pos_per_dim * cos_sin_stride_pos
        else:
            pos = tl.load(pos_ptr + token_id)
            cos_base = pos * cos_sin_stride_pos

        cos_vals = tl.load(cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0).to(tl.float32)
        sin_vals = tl.load(sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0).to(tl.float32)

        gather_idx = tl.where(
            first_half_mask,
            d_offs + ROTARY_DIM_HALF,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
        )
        k_gathered_raw = tl.load(k_ptr + k_in_offset + gather_idx).to(tl.float32)
        # ===== BF16 (1 + kw_gathered) =====
        kw_g_bf16 = tl.load(kw_ptr + gather_idx)
        kw_g_plus1 = (kw_g_bf16 + 1.0).to(tl.float32)
        k_gathered_normed = k_gathered_raw * tl.math.rsqrt(k_variance + eps) * kw_g_plus1
        k_rot = tl.where(first_half_mask, -k_gathered_normed, k_gathered_normed)
        k_rot = tl.where(rot_mask, k_rot, 0.0)

        k_roped = k_normed * cos_vals + k_rot * sin_vals

        k_out_offset = token_id * (num_kv_heads * BLOCK_D) + kv_h * BLOCK_D
        tl.store(k_out_ptr + k_out_offset + d_offs, k_roped.to(k_out_ptr.dtype.element_ty))

        v_in_offset = token_id * v_stride_t + kv_h * BLOCK_D
        v = tl.load(v_ptr + v_in_offset + d_offs)

        slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
        if slot >= 0:
            block_idx = slot // BLOCK_SIZE
            slot_in_block = slot % BLOCK_SIZE

            if IS_FP8:
                k_abs_max = tl.max(tl.abs(k_roped), axis=0)
                k_scale = k_abs_max / 240.0
                k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
                k_quant = (k_roped / k_scale).to(k_cache_ptr.dtype.element_ty)

                tl.store(
                    k_scale_ptr
                    + block_idx * ks_stride_block
                    + kv_h * ks_stride_head
                    + slot_in_block,
                    k_scale,
                )
            else:
                k_quant = k_roped.to(k_cache_ptr.dtype.element_ty)

            k_quant_2d = tl.reshape(k_quant, (BLOCK_D // X_SIZE, X_SIZE))
            dx_offs = tl.arange(0, BLOCK_D // X_SIZE).to(tl.int64)
            x_offs = tl.arange(0, X_SIZE).to(tl.int64)
            k_cache_ptrs = (
                k_cache_ptr
                + block_idx * kc_stride_block
                + kv_h * kc_stride_head
                + dx_offs[:, None] * kc_stride_dx
                + slot_in_block * kc_stride_slot
                + x_offs[None, :] * kc_stride_x
            )
            tl.store(k_cache_ptrs, k_quant_2d)

            if IS_FP8:
                v_f32 = v.to(tl.float32)
                v_abs_max = tl.max(tl.abs(v_f32), axis=0)
                v_scale = v_abs_max / 240.0
                v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
                v_quant = (v_f32 / v_scale).to(v_cache_ptr.dtype.element_ty)

                tl.store(
                    v_scale_ptr
                    + block_idx * vs_stride_block
                    + kv_h * vs_stride_head
                    + slot_in_block,
                    v_scale,
                )
            else:
                v_quant = v.to(v_cache_ptr.dtype.element_ty)

            slot_chunk = slot_in_block // X_SIZE
            x_off = slot_in_block % X_SIZE
            v_cache_ptrs = (
                v_cache_ptr
                + block_idx * vc_stride_block
                + kv_h * vc_stride_head
                + slot_chunk * vc_stride_sc
                + d_offs.to(tl.int64) * vc_stride_d
                + x_off * vc_stride_x
            )
            tl.store(v_cache_ptrs, v_quant)


def _patched_triton_fused_norm_rope_cache(
    q, k, v, positions,
    q_norm, k_norm, rotary_emb,
    num_heads, num_kv_heads, head_dim,
    k_cache, v_cache, k_scale, v_scale,
    slot_mapping, kv_cache_dtype,
):
    T = q.shape[0]
    eps = q_norm.variance_epsilon
    rotary_dim = rotary_emb.rotary_dim

    cos_cache = rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
    sin_cache = rotary_emb.sin_cache.squeeze(-2).squeeze(-2)

    is_fp8 = kv_cache_dtype == "fp8"

    block_size = k_cache.shape[3]
    x_size = k_cache.shape[4]

    is_mrope = positions.ndim == 2
    mrope_section = getattr(rotary_emb, "mrope_section", None)
    if is_mrope:
        assert mrope_section is not None
        s0 = mrope_section[0]
        s1 = s0 + mrope_section[1]
        pos_stride_row = positions.stride(0)
    else:
        s0 = 0
        s1 = 0
        pos_stride_row = 0

    _PATCH_COUNTERS["triton_kernel_calls"] += 1
    _PATCH_COUNTERS["triton_kernel_unique_shapes"].add(
        (tuple(q.shape), tuple(k.shape), num_heads, num_kv_heads, head_dim)
    )

    q_out = q.new_empty((T, num_heads * head_dim), dtype=q.dtype)
    k_out = k.new_empty((T, num_kv_heads * head_dim), dtype=k.dtype)

    total_heads = num_heads + num_kv_heads
    grid = (T * total_heads,)

    _fused_qkv_norm_rope_cache_kernel_bf16[grid](
        q, q.stride(0),
        k, k.stride(0),
        v, v.stride(0),
        q_out, k_out,
        q_norm.weight, k_norm.weight,
        cos_cache, sin_cache,
        cos_cache.stride(0),
        positions,
        k_cache, v_cache,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        k_cache.stride(3), k_cache.stride(4),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        v_cache.stride(3), v_cache.stride(4),
        k_scale if k_scale is not None else q,
        v_scale if v_scale is not None else q,
        k_scale.stride(0) if k_scale is not None and k_scale.dim() >= 1 else 0,
        k_scale.stride(1) if k_scale is not None and k_scale.dim() >= 2 else 0,
        v_scale.stride(0) if v_scale is not None and v_scale.dim() >= 1 else 0,
        v_scale.stride(1) if v_scale is not None and v_scale.dim() >= 2 else 0,
        slot_mapping,
        pos_stride_row,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        eps=eps,
        BLOCK_SIZE=block_size,
        X_SIZE=x_size,
        BLOCK_D=head_dim,
        ROTARY_DIM=rotary_dim,
        ROTARY_DIM_HALF=rotary_dim // 2,
        IS_FP8=is_fp8,
        MROPE_S0=s0,
        MROPE_S1=s1,
        IS_MROPE=is_mrope,
    )

    return q_out, k_out


_tfqnrc.triton_fused_norm_rope_cache = _patched_triton_fused_norm_rope_cache

# Also patch the import inside attention_mha (which captured the original at import time).
from atom.model_ops import attention_mha as _amha  # noqa: E402
if hasattr(_amha, "triton_fused_norm_rope_cache"):
    _amha.triton_fused_norm_rope_cache = _patched_triton_fused_norm_rope_cache

print("[hack] Patched triton_fused_norm_rope_cache with bf16-(1+w) variant")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
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
    summary = {
        "engine": "atom-full-bf16-triton",
        "results": results,
        "patch_counters": {
            "gemma_rms_static_calls": _PATCH_COUNTERS["gemma_rms_static_calls"],
            "gemma_rms_static_unique_shapes": [
                list(s) for s in _PATCH_COUNTERS["gemma_rms_static_unique_shapes"]
            ],
            "triton_kernel_calls": _PATCH_COUNTERS["triton_kernel_calls"],
            "triton_kernel_unique_shapes": [
                list(s) for s in _PATCH_COUNTERS["triton_kernel_unique_shapes"]
            ],
        },
    }
    print(json.dumps(summary, indent=2))
    print("__SUMMARY_END__")


if __name__ == "__main__":
    main()
