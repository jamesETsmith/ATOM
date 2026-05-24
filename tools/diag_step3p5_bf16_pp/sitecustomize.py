"""sitecustomize.py — bf16 GemmaRMSNorm patch for every spawned ATOM worker.

Activates when ATOM_DIAG_BF16=1. Patches the actual module objects
immediately after they finish loading, using a sys.meta_path hook that
wraps the default loader.
"""
from __future__ import annotations

import atexit
import json
import os
import sys

_ENABLED = os.environ.get("ATOM_DIAG_BF16", "") == "1"
_DEBUG = os.environ.get("ATOM_DIAG_BF16_DEBUG", "") == "1"

if _DEBUG:
    sys.stderr.write(
        f"[bf16-diag] sitecustomize loaded pid={os.getpid()} enabled={_ENABLED}\n"
    )
    sys.stderr.flush()

if _ENABLED:
    OUT_BASE = os.environ.get("ATOM_DIAG_BF16_OUT", "/tmp/atom_diag_bf16")

    _COUNTERS = {
        "pid": os.getpid(),
        "gemma_rms_static_calls": 0,
        "triton_kernel_calls": 0,
    }

    _TARGETS = {
        "atom.model_ops.layernorm",
        "atom.model_ops.triton_fused_qkv_norm_rope_cache",
    }
    _done = set()

    _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _hooked_import(name, *args, **kwargs):
        result = _real_import(name, *args, **kwargs)
        # Check if any of our targets just appeared in sys.modules.
        for target in _TARGETS - _done:
            if target in sys.modules:
                _done.add(target)
                _apply_patch(target)
        return result

    def _apply_patch(target):
        if target == "atom.model_ops.layernorm":
            _do_layernorm_patch()
        elif target == "atom.model_ops.triton_fused_qkv_norm_rope_cache":
            _do_triton_patch()

    def _do_layernorm_patch():
        import torch
        _ln = sys.modules["atom.model_ops.layernorm"]

        @staticmethod
        def _bf16_static(weight, variance_epsilon, x, residual):
            _COUNTERS["gemma_rms_static_calls"] += 1
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
            gemma_weight = weight + 1.0  # bf16 add
            out = x_norm * gemma_weight
            return out if residual is None else (out, residual)

        _ln.GemmaRMSNorm.forward_static = _bf16_static

        def _bypass_cuda(self, x, residual=None):
            return self.forward_native(x, residual)
        _ln.GemmaRMSNorm.forward_cuda = _bypass_cuda

        if _DEBUG:
            sys.stderr.write(f"[bf16-diag] pid={os.getpid()} PATCHED GemmaRMSNorm\n")
            sys.stderr.flush()

    def _do_triton_patch():
        import torch
        import triton
        import triton.language as tl

        _mod = sys.modules["atom.model_ops.triton_fused_qkv_norm_rope_cache"]

        @triton.jit
        def _bf16_kernel(
            q_ptr, q_stride_t, k_ptr, k_stride_t, v_ptr, v_stride_t,
            q_out_ptr, k_out_ptr, qw_ptr, kw_ptr,
            cos_cache_ptr, sin_cache_ptr, cos_sin_stride_pos, pos_ptr,
            k_cache_ptr, v_cache_ptr,
            kc_stride_block, kc_stride_head, kc_stride_dx, kc_stride_slot, kc_stride_x,
            vc_stride_block, vc_stride_head, vc_stride_sc, vc_stride_d, vc_stride_x,
            k_scale_ptr, v_scale_ptr,
            ks_stride_block, ks_stride_head, vs_stride_block, vs_stride_head,
            slot_mapping_ptr, pos_stride_row,
            num_heads: tl.constexpr, num_kv_heads: tl.constexpr, eps: tl.constexpr,
            BLOCK_SIZE: tl.constexpr, X_SIZE: tl.constexpr, BLOCK_D: tl.constexpr,
            ROTARY_DIM: tl.constexpr, ROTARY_DIM_HALF: tl.constexpr,
            IS_FP8: tl.constexpr,
            MROPE_S0: tl.constexpr = 0, MROPE_S1: tl.constexpr = 0,
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
                # BF16 (1+w)
                qw_bf16 = tl.load(qw_ptr + d_offs)
                qw_plus1 = (qw_bf16 + 1.0).to(tl.float32)
                q_normed = q_normed * qw_plus1

                rot_mask = d_offs < ROTARY_DIM
                first_half_mask = d_offs < ROTARY_DIM_HALF
                d_cos_idx = tl.where(first_half_mask, d_offs,
                    tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)))
                if IS_MROPE:
                    pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
                    pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
                    pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
                    pos_per_dim = tl.where(d_cos_idx < MROPE_S0, pos_t,
                        tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w))
                    cos_base = pos_per_dim * cos_sin_stride_pos
                else:
                    pos = tl.load(pos_ptr + token_id)
                    cos_base = pos * cos_sin_stride_pos
                cos_vals = tl.load(cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0).to(tl.float32)
                sin_vals = tl.load(sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0).to(tl.float32)
                gather_idx = tl.where(first_half_mask, d_offs + ROTARY_DIM_HALF,
                    tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs))
                q_gathered_raw = tl.load(q_ptr + q_in_offset + gather_idx).to(tl.float32)
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
                kw_bf16 = tl.load(kw_ptr + d_offs)
                kw_plus1 = (kw_bf16 + 1.0).to(tl.float32)
                k_normed = k_normed * kw_plus1
                rot_mask = d_offs < ROTARY_DIM
                first_half_mask = d_offs < ROTARY_DIM_HALF
                d_cos_idx = tl.where(first_half_mask, d_offs,
                    tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)))
                if IS_MROPE:
                    pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
                    pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
                    pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
                    pos_per_dim = tl.where(d_cos_idx < MROPE_S0, pos_t,
                        tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w))
                    cos_base = pos_per_dim * cos_sin_stride_pos
                else:
                    pos = tl.load(pos_ptr + token_id)
                    cos_base = pos * cos_sin_stride_pos
                cos_vals = tl.load(cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0).to(tl.float32)
                sin_vals = tl.load(sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0).to(tl.float32)
                gather_idx = tl.where(first_half_mask, d_offs + ROTARY_DIM_HALF,
                    tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs))
                k_gathered_raw = tl.load(k_ptr + k_in_offset + gather_idx).to(tl.float32)
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
                        tl.store(k_scale_ptr + block_idx * ks_stride_block + kv_h * ks_stride_head + slot_in_block, k_scale)
                    else:
                        k_quant = k_roped.to(k_cache_ptr.dtype.element_ty)
                    k_quant_2d = tl.reshape(k_quant, (BLOCK_D // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_cache_ptrs = (k_cache_ptr + block_idx * kc_stride_block + kv_h * kc_stride_head
                        + dx_offs[:, None] * kc_stride_dx + slot_in_block * kc_stride_slot + x_offs[None, :] * kc_stride_x)
                    tl.store(k_cache_ptrs, k_quant_2d)
                    if IS_FP8:
                        v_f32 = v.to(tl.float32)
                        v_abs_max = tl.max(tl.abs(v_f32), axis=0)
                        v_scale = v_abs_max / 240.0
                        v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
                        v_quant = (v_f32 / v_scale).to(v_cache_ptr.dtype.element_ty)
                        tl.store(v_scale_ptr + block_idx * vs_stride_block + kv_h * vs_stride_head + slot_in_block, v_scale)
                    else:
                        v_quant = v.to(v_cache_ptr.dtype.element_ty)
                    slot_chunk = slot_in_block // X_SIZE
                    x_off = slot_in_block % X_SIZE
                    v_cache_ptrs = (v_cache_ptr + block_idx * vc_stride_block + kv_h * vc_stride_head
                        + slot_chunk * vc_stride_sc + d_offs.to(tl.int64) * vc_stride_d + x_off * vc_stride_x)
                    tl.store(v_cache_ptrs, v_quant)

        def _patched_wrapper(q, k, v, positions, q_norm, k_norm, rotary_emb,
                             num_heads, num_kv_heads, head_dim,
                             k_cache, v_cache, k_scale, v_scale,
                             slot_mapping, kv_cache_dtype):
            _COUNTERS["triton_kernel_calls"] += 1
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
                s0 = s1 = pos_stride_row = 0
            q_out = q.new_empty((T, num_heads * head_dim), dtype=q.dtype)
            k_out = k.new_empty((T, num_kv_heads * head_dim), dtype=k.dtype)
            grid = (T * (num_heads + num_kv_heads),)
            _bf16_kernel[grid](
                q, q.stride(0), k, k.stride(0), v, v.stride(0),
                q_out, k_out, q_norm.weight, k_norm.weight,
                cos_cache, sin_cache, cos_cache.stride(0), positions,
                k_cache, v_cache,
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), k_cache.stride(4),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3), v_cache.stride(4),
                k_scale if k_scale is not None else q, v_scale if v_scale is not None else q,
                k_scale.stride(0) if k_scale is not None and k_scale.dim() >= 1 else 0,
                k_scale.stride(1) if k_scale is not None and k_scale.dim() >= 2 else 0,
                v_scale.stride(0) if v_scale is not None and v_scale.dim() >= 1 else 0,
                v_scale.stride(1) if v_scale is not None and v_scale.dim() >= 2 else 0,
                slot_mapping, pos_stride_row,
                num_heads=num_heads, num_kv_heads=num_kv_heads, eps=eps,
                BLOCK_SIZE=block_size, X_SIZE=x_size, BLOCK_D=head_dim,
                ROTARY_DIM=rotary_dim, ROTARY_DIM_HALF=rotary_dim // 2,
                IS_FP8=is_fp8, MROPE_S0=s0, MROPE_S1=s1, IS_MROPE=is_mrope,
            )
            return q_out, k_out

        _mod.triton_fused_norm_rope_cache = _patched_wrapper
        if _DEBUG:
            sys.stderr.write(f"[bf16-diag] pid={os.getpid()} PATCHED triton kernel\n")
            sys.stderr.flush()

    import builtins as _builtins
    _real_import = _builtins.__import__

    def _hooked_import(name, *args, **kwargs):
        result = _real_import(name, *args, **kwargs)
        for target in list(_TARGETS - _done):
            if target in sys.modules:
                _done.add(target)
                try:
                    _apply_patch(target)
                except Exception as e:
                    sys.stderr.write(f"[bf16-diag] patch {target} err: {e}\n")
                    import traceback; traceback.print_exc()
                    sys.stderr.flush()
        return result

    _builtins.__import__ = _hooked_import

    def _dump_counters():
        out_path = f"{OUT_BASE}.{os.getpid()}.json"
        try:
            with open(out_path, "w") as f:
                json.dump(_COUNTERS, f, indent=2)
            if _DEBUG:
                sys.stderr.write(f"[bf16-diag] pid={os.getpid()} dumped -> {out_path}\n")
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[bf16-diag] dump err: {e}\n")

    atexit.register(_dump_counters)
