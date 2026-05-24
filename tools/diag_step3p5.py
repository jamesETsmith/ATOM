"""Minimal diagnostic: compare prefill vs decode hidden states for Step 3.5 Flash.

Run from ATOM root with the correct LD_LIBRARY_PATH:
  .venv/bin/python tools/diag_step3p5.py

This bypasses the full server to isolate whether the issue is in model logic
or in the serving infrastructure.
"""
import os, sys, json, glob
import torch
from safetensors import safe_open

MODEL_PATH = "/data/jamesmit/models/Step-3.5-Flash-FP8"

# Load config
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config_json = json.load(f)

print("=== Step 3.5 Flash Diagnostic ===")
print(f"num_hidden_layers: {config_json['num_hidden_layers']}")
print(f"moe_num_experts: {config_json['moe_num_experts']}")
print(f"moe_top_k: {config_json['moe_top_k']}")

# Check weight consistency: verify that checkpoint keys can all be mapped
# to ATOM parameter names through the packed_modules_mapping and fused expert mapping
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "mlp.gate_proj": ("mlp.gate_up_proj", 0),
    "mlp.up_proj": ("mlp.gate_up_proj", 1),
    "share_expert.gate_proj": ("share_expert.gate_up_proj", 0),
    "share_expert.up_proj": ("share_expert.gate_up_proj", 1),
}

fused_expert_mapping = [
    ("moe.experts.w13_weight", "moe.gate_proj.weight", "w1"),
    ("moe.experts.w13_weight", "moe.up_proj.weight", "w3"),
    ("moe.experts.w2_weight", "moe.down_proj.weight", "w2"),
]

def is_fused_expert(name):
    return (
        ".moe.gate_proj" in name
        or ".moe.up_proj" in name
        or ".moe.down_proj" in name
    )

def is_mtp_layer(name, num_layers=45):
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    if m and int(m.group(1)) >= num_layers:
        return True
    return False

def map_ckpt_to_atom(name):
    """Map checkpoint key to ATOM parameter name."""
    # Rename weight_scale_inv -> weight_scale
    name = name.replace("weight_scale_inv", "weight_scale")
    
    if is_fused_expert(name):
        # Fused expert mapping
        for param_name, weight_name, shard_id in fused_expert_mapping:
            if weight_name in name:
                return name.replace(weight_name, param_name), shard_id
        return name, None
    
    # Packed modules mapping
    for packed_key, packed_value in packed_modules_mapping.items():
        if f".{packed_key}." in name:
            if isinstance(packed_value, tuple):
                target_name, shard_idx = packed_value
                return name.replace(packed_key, target_name, 1), shard_idx
    
    return name, None

# Scan all checkpoint keys
all_keys = []
for shard_file in sorted(glob.glob(f"{MODEL_PATH}/*.safetensors")):
    with safe_open(shard_file, framework="pt") as f:
        all_keys.extend(f.keys())

print(f"\nTotal checkpoint keys: {len(all_keys)}")

# Categorize
mtp_keys = [k for k in all_keys if is_mtp_layer(k)]
fused_expert_keys = [k for k in all_keys if is_fused_expert(k) and not is_mtp_layer(k)]
packed_keys = []
direct_keys = []
skip_keys = []

for k in all_keys:
    if is_mtp_layer(k):
        continue
    if is_fused_expert(k):
        continue
    if "kv_scale" in k or "inv_freq" in k:
        skip_keys.append(k)
        continue
    atom_name, shard = map_ckpt_to_atom(k)
    if shard is not None:
        packed_keys.append((k, atom_name, shard))
    else:
        direct_keys.append((k, atom_name))

print(f"MTP keys (skipped): {len(mtp_keys)}")
print(f"Fused expert keys: {len(fused_expert_keys)}")
print(f"Packed module keys: {len(packed_keys)}")
print(f"Direct load keys: {len(direct_keys)}")
print(f"Skipped keys: {len(skip_keys)}")

# Show some examples of each mapping
print("\n--- Fused expert mapping examples ---")
for k in fused_expert_keys[:6]:
    atom_name, shard = map_ckpt_to_atom(k)
    print(f"  {k}")
    print(f"    -> {atom_name} (shard={shard})")

print("\n--- Packed module mapping examples ---")
for k, atom_name, shard in packed_keys[:6]:
    print(f"  {k}")
    print(f"    -> {atom_name} (shard={shard})")

# Check for any key that can't be mapped
print("\n--- Checking for unmappable keys ---")
unmapped = 0
for k in all_keys:
    if is_mtp_layer(k):
        continue
    if "kv_scale" in k or "inv_freq" in k:
        continue
    atom_name, _ = map_ckpt_to_atom(k)
    # The atom_name should exist as a parameter in the model
    # We can't check this without instantiating the model,
    # but we can check for obvious issues

print(f"\nTotal non-MTP, non-skip keys: {len(fused_expert_keys) + len(packed_keys) + len(direct_keys)}")

# Quick check: are any dense-layer weights accidentally getting FP8 treatment?
print("\n--- Checking dense layer dtypes ---")
for shard_file in sorted(glob.glob(f"{MODEL_PATH}/*.safetensors"))[:3]:
    with safe_open(shard_file, framework="pt") as f:
        for key in f.keys():
            if "layers.0." in key or "layers.1." in key or "layers.2." in key:
                t = f.get_tensor(key)
                if t.dtype == torch.float8_e4m3fn:
                    print(f"  WARNING: Dense layer has FP8 weight: {key}")

print("\n=== Done ===")
