"""Quick diagnostic for Step 3.5 Flash weight loading.

Run from the ATOM root with:
  LD_LIBRARY_PATH=... HIP_VISIBLE_DEVICES=0 .venv/bin/python tools/check_step3p5_weights.py
"""
import sys
import os
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

import torch

def main():
    model_path = "/data/jamesmit/models/Step-3.5-Flash-FP8"

    from safetensors import safe_open
    import glob

    # Load checkpoint keys and a few weight stats
    files = sorted(glob.glob(f"{model_path}/*.safetensors"))

    # Check a specific layer's MoE gate_proj weight
    print("=== Checkpoint Expert Weight Stats ===")
    for f in files[:5]:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if "layers.5.moe.gate_proj.weight" == k.split("model.")[-1]:
                    t = sf.get_tensor(k)
                    print(f"  {k}: shape={t.shape}, dtype={t.dtype}, "
                          f"abs_mean={t.float().abs().mean():.6f}, "
                          f"max={t.float().abs().max():.6f}")
                    break
            else:
                continue
            break

    print("\n=== Loading ATOM model ===")
    from atom.config import Config
    from atom.model_config.step3p5 import Step3p5Config
    from transformers import AutoConfig

    # Load config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    print(f"  Config type: {type(hf_config).__name__}")
    print(f"  num_experts: {hf_config.num_experts}")
    print(f"  routed_scaling_factor: {hf_config.routed_scaling_factor}")

    # Check model parameter names
    print("\n=== Model Parameter Names (first MoE layer) ===")
    # We can't easily instantiate the full model without the engine,
    # but we can check what the loader would see.

    # Instead, let's check if the loaded weights in a running server
    # have non-zero expert weights by examining checkpoint vs scale stats
    print("\n=== Checking weight_scale_inv values ===")
    for f in files[:5]:
        with safe_open(f, framework="pt") as sf:
            for k in sorted(sf.keys()):
                if "layers.5.moe" in k and "scale" in k:
                    t = sf.get_tensor(k)
                    print(f"  {k}: shape={t.shape}, dtype={t.dtype}, "
                          f"mean={t.mean():.6f}, max={t.max():.6f}, min={t.min():.6f}")
                    break
            else:
                continue
            break

    # Check non-MoE weights for comparison
    print("\n=== Non-MoE (layer 0) weight stats ===")
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sorted(sf.keys()):
                if "layers.0." in k and "weight" in k:
                    t = sf.get_tensor(k)
                    print(f"  {k}: shape={t.shape}, dtype={t.dtype}, "
                          f"abs_mean={t.float().abs().mean():.6f}")
            break  # Only first shard

    # Check the FP8 expert weights are actually non-trivial
    print("\n=== FP8 Expert Weight Histogram (layer 5, gate_proj, expert 0) ===")
    for f in files:
        with safe_open(f, framework="pt") as sf:
            if "model.layers.5.moe.gate_proj.weight" in sf.keys():
                t = sf.get_tensor("model.layers.5.moe.gate_proj.weight")
                expert0 = t[0]  # First expert
                vals = expert0.float()
                print(f"  Shape: {expert0.shape}, dtype: {expert0.dtype}")
                print(f"  abs_mean: {vals.abs().mean():.6f}")
                print(f"  zeros: {(vals == 0).sum().item()} / {vals.numel()}")
                print(f"  non-zero: {(vals != 0).sum().item()} / {vals.numel()}")
                # Show distribution
                abs_vals = vals.abs()
                for threshold in [0.001, 0.01, 0.1, 0.5, 1.0]:
                    count = (abs_vals > threshold).sum().item()
                    print(f"    |val| > {threshold}: {count}")
                break

if __name__ == "__main__":
    main()
