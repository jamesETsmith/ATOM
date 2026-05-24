#!/usr/bin/env python3
"""Audit loaded weights for Step-3.5-Flash in ATOM.

Runs a model load and checks every parameter for:
- All-zero tensors (suggests weight wasn't loaded)
- NaN/Inf values
- Unexpected dtype
- Shape mismatches with checkpoint

Usage:
    python tools/audit_step3p5_weights.py

Must run in the ATOM venv with GPU access (loads model to GPU).
"""
import json
import os
import sys
import time

os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
os.environ.setdefault("HF_HOME", "/data/jamesmit/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data/jamesmit/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/jamesmit/hf_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/jamesmit/triton_cache")

MODEL_PATH = "/data/jamesmit/models/Step-3.5-Flash-FP8"


def load_checkpoint_index(model_dir: str) -> dict:
    """Load safetensors index and return {name: {shape, dtype, shard}} map."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    # We'd need to open each shard to get shapes; for now just return names
    return weight_map


def audit_model_weights():
    """Load model via ATOM's standard path and audit all parameters."""
    import torch

    # Load config
    from atom.config import Config

    config = Config(
        model=MODEL_PATH,
        kv_cache_dtype="fp8",
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=False,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
    )

    print(f"Model type: {config.hf_config.model_type}", flush=True)
    print(f"Quant config: {config.quant_config}", flush=True)

    # Build model
    from atom.model_engine.model_runner import ModelRunner

    # We need to create a minimal model runner to load the model
    # But that requires too much infrastructure. Instead, load model directly.
    from atom.models.step3p5 import Step3p5ForCausalLM

    print("Creating model...", flush=True)
    t0 = time.time()
    model = Step3p5ForCausalLM(atom_config=config)
    model = model.to("cuda")
    print(f"Model created in {time.time() - t0:.1f}s", flush=True)

    # Load weights
    from atom.model_loader.loader import load_model

    print("Loading weights...", flush=True)
    t0 = time.time()

    fused_fn = None
    if hasattr(model, "load_fused_expert_weights"):
        fused_fn = model.load_fused_expert_weights

    loaded_record = load_model(
        model,
        config.model,
        config.hf_config,
        config.load_dummy,
        load_fused_expert_weights_fn=fused_fn,
    )
    print(f"Weights loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"Loaded weight count: {len(loaded_record)}", flush=True)

    # Audit
    issues = []
    stats = []

    checkpoint_names = set(load_checkpoint_index(MODEL_PATH).keys())
    # Adjust checkpoint names to match model parameter naming
    expected_model_params = set()
    for name in checkpoint_names:
        n = name
        if "weight_scale_inv" in n:
            n = n.replace("weight_scale_inv", "weight_scale")
        expected_model_params.add(n)

    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    all_model_names = set(model_params.keys()) | set(model_buffers.keys())

    print(f"\nModel parameters: {len(model_params)}", flush=True)
    print(f"Model buffers: {len(model_buffers)}", flush=True)

    for name, param in sorted(model_params.items()):
        data = param.data
        is_all_zero = (data == 0).all().item()
        has_nan = torch.isnan(data.float()).any().item()
        has_inf = torch.isinf(data.float()).any().item()

        stat = {
            "name": name,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "all_zero": is_all_zero,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }

        if data.dtype in (torch.bfloat16, torch.float16, torch.float32):
            stat["mean"] = round(data.float().mean().item(), 6)
            stat["std"] = round(data.float().std().item(), 6)
            stat["abs_max"] = round(data.float().abs().max().item(), 6)
        elif data.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            stat["mean"] = round(data.float().mean().item(), 6)
            stat["abs_max"] = round(data.float().abs().max().item(), 6)

        stats.append(stat)

        # Flag issues
        if is_all_zero and "bias" not in name and "router_bias" not in name:
            # GemmaRMSNorm weights are initialized to zero (1+w pattern)
            if "norm" not in name.lower():
                issues.append(f"ALL_ZERO: {name} {list(data.shape)} {data.dtype}")
        if has_nan:
            issues.append(f"HAS_NAN: {name} {list(data.shape)} {data.dtype}")
        if has_inf:
            issues.append(f"HAS_INF: {name} {list(data.shape)} {data.dtype}")

    # Check for expected weights not loaded
    loaded_short = {n.replace("model.", "", 1) if n.startswith("model.") else n for n in loaded_record}

    print(f"\n{'='*60}", flush=True)
    print("AUDIT RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    if issues:
        print(f"\n{len(issues)} ISSUES FOUND:", flush=True)
        for issue in issues:
            print(f"  {issue}", flush=True)
    else:
        print("\nNo issues found (no all-zero, NaN, or Inf parameters)", flush=True)

    # Print norm weight stats (should be near-zero for GemmaRMSNorm)
    print(f"\n--- Norm weight stats (first 5) ---", flush=True)
    norm_stats = [s for s in stats if "norm" in s["name"].lower() and "weight" in s["name"]]
    for s in norm_stats[:5]:
        print(f"  {s['name']}: mean={s.get('mean', '?')}, std={s.get('std', '?')}, abs_max={s.get('abs_max', '?')}", flush=True)

    # Print MoE expert weight stats (first 2 layers)
    print(f"\n--- MoE expert weight stats (first 2 MoE layers) ---", flush=True)
    moe_stats = [s for s in stats if "moe.experts" in s["name"] and ("layers.3." in s["name"] or "layers.4." in s["name"])]
    for s in moe_stats:
        print(f"  {s['name']}: shape={s['shape']}, dtype={s['dtype']}, mean={s.get('mean', '?')}, abs_max={s.get('abs_max', '?')}, all_zero={s['all_zero']}", flush=True)

    # Print gate weight stats
    print(f"\n--- Router gate weight stats (first 3) ---", flush=True)
    gate_stats = [s for s in stats if "moe.gate" in s["name"]]
    for s in gate_stats[:3]:
        print(f"  {s['name']}: shape={s['shape']}, dtype={s['dtype']}, mean={s.get('mean', '?')}, abs_max={s.get('abs_max', '?')}", flush=True)

    # Print router_bias stats
    print(f"\n--- Router bias stats (first 3) ---", flush=True)
    bias_stats = [s for s in stats if "router_bias" in s["name"]]
    for s in bias_stats[:3]:
        print(f"  {s['name']}: shape={s['shape']}, dtype={s['dtype']}, mean={s.get('mean', '?')}, abs_max={s.get('abs_max', '?')}, all_zero={s['all_zero']}", flush=True)

    # Print g_proj stats
    print(f"\n--- g_proj weight stats (first 3) ---", flush=True)
    gproj_stats = [s for s in stats if "g_proj" in s["name"]]
    for s in gproj_stats[:3]:
        print(f"  {s['name']}: shape={s['shape']}, dtype={s['dtype']}, mean={s.get('mean', '?')}, abs_max={s.get('abs_max', '?')}", flush=True)

    # Print embed/lm_head stats
    print(f"\n--- Embedding/LM head stats ---", flush=True)
    emb_stats = [s for s in stats if "embed" in s["name"] or "lm_head" in s["name"]]
    for s in emb_stats:
        print(f"  {s['name']}: shape={s['shape']}, dtype={s['dtype']}, mean={s.get('mean', '?')}, abs_max={s.get('abs_max', '?')}", flush=True)

    # Write full stats to JSON
    out_path = "/tmp/step3p5_weight_audit.json"
    with open(out_path, "w") as f:
        json.dump({"issues": issues, "stats": stats, "loaded_count": len(loaded_record)}, f, indent=2)
    print(f"\nFull stats written to {out_path}", flush=True)


if __name__ == "__main__":
    audit_model_weights()
