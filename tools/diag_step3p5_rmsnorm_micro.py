"""Microtest: feed the L0 input layernorm with a synthetic input and
compare ATOM's GemmaRMSNorm forward path against:
  1. A bit-exact FP32 PyTorch reference.
  2. AITER's rms_norm with precomputed (1+w) weight (== SGLang HIP path).

Loads the actual `model.layers.0.input_layernorm.weight` from the
Step-3.5-Flash-FP8 checkpoint and uses an embedding-like input.
"""

from __future__ import annotations

import os
import sys

import torch

os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

MODEL = "/data/jamesmit/models/Step-3.5-Flash-FP8"


def _load_weight_and_embed():
    from safetensors import safe_open
    import json
    import glob

    # Find the safetensors shard that contains the L0 input_layernorm weight
    # by reading the index.
    idx_path = os.path.join(MODEL, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]
    target = "model.layers.0.input_layernorm.weight"
    if target not in weight_map:
        # Try alternates
        for k in weight_map:
            if "layers.0" in k and "input_layernorm" in k:
                target = k
                break
    shard = weight_map[target]
    embed_target = "model.embed_tokens.weight"
    embed_shard = weight_map[embed_target]

    with safe_open(os.path.join(MODEL, shard), framework="pt") as f:
        w = f.get_tensor(target)
    with safe_open(os.path.join(MODEL, embed_shard), framework="pt") as f:
        embed = f.get_tensor(embed_target)
    return w, embed


def _embed_prompt():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = tok.encode("What is 2+2?")
    return torch.tensor(ids, dtype=torch.long), ids


def _fp32_reference(x_bf16, w_bf16, eps):
    x = x_bf16.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    x = x * (1.0 + w_bf16.float())
    return x.to(torch.bfloat16)


def main():
    device = "cuda"
    torch.set_grad_enabled(False)

    print("Loading weights and embedding...")
    w, embed = _load_weight_and_embed()
    ids, id_list = _embed_prompt()
    print(f"  prompt tokens: {id_list}")
    print(f"  embed shape: {embed.shape} dtype: {embed.dtype}")
    print(f"  layernorm weight shape: {w.shape} dtype: {w.dtype}")

    embed = embed.to(device)
    w = w.to(device)
    x = embed[ids.to(device)]  # (8, 4096)
    print(f"  L0 input shape: {x.shape} dtype: {x.dtype} norm: {x.float().norm().item():.6f}")
    print(f"  expected from capture (norm=2.004599094390869)")

    eps = 1e-5
    print(f"\n=== Running each variant with eps={eps} ===")

    # Variant 1: ATOM's GemmaRMSNorm.forward_native (== forward_static eager FP32)
    sys.path.insert(0, "/home/AMD/jamesmit/apps/ATOM")
    from atom.model_ops.layernorm import GemmaRMSNorm
    norm = GemmaRMSNorm(4096, eps=eps).to(device)
    norm.weight.data.copy_(w)
    out_atom = norm.forward_native(x.clone())
    print(f"\n[ATOM forward_native]")
    print(f"  out norm: {out_atom.float().norm().item():.6f}")
    print(f"  last_tok[:5]: {out_atom[-1, :5].float().tolist()}")

    # Variant 2: pure FP32 reference
    out_ref = _fp32_reference(x, w, eps)
    print(f"\n[Pure FP32 reference]")
    print(f"  out norm: {out_ref.float().norm().item():.6f}")
    print(f"  last_tok[:5]: {out_ref[-1, :5].float().tolist()}")
    print(f"  vs ATOM: bit-exact={torch.equal(out_atom, out_ref)}, "
          f"max_abs_diff={(out_atom.float() - out_ref.float()).abs().max().item():.6e}")

    # Variant 3: AITER rms_norm with precomputed (1+w)
    try:
        from aiter import rmsnorm2d_fwd as rms_norm
        gemma_w = (w.float() + 1.0).to(w.dtype)
        out_aiter = rms_norm(x.clone(), gemma_w, eps)
        print(f"\n[AITER rmsnorm2d_fwd (SGLang HIP path)]")
        print(f"  out norm: {out_aiter.float().norm().item():.6f}")
        print(f"  last_tok[:5]: {out_aiter[-1, :5].float().tolist()}")
        print(f"  vs ATOM ref:  max_abs_diff={(out_atom.float() - out_aiter.float()).abs().max().item():.6e}")
        print(f"  vs FP32 ref:  max_abs_diff={(out_ref.float() - out_aiter.float()).abs().max().item():.6e}")
    except Exception as exc:
        print(f"\n[AITER rmsnorm2d_fwd] FAILED: {exc!r}")

    # SGLang capture says norm should be 139.6490 (post-norm shape [8, 4096])
    # ATOM capture says norm should be 140.0497
    # Pure FP32 should match one of them.
    print(f"\n=== Captured norms for reference ===")
    print(f"  SGLang attn_input (post-norm) norm: 139.6490")
    print(f"  ATOM   attn_input (post-norm) norm: 140.0497")
    print(f"  This run's outputs:")
    print(f"    ATOM   path: {out_atom.float().norm().item():.4f}")
    print(f"    FP32   ref : {out_ref.float().norm().item():.4f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
