#!/usr/bin/env python3
"""Diagnostic: compare ATOM & vLLM embedding outputs for Step3p5.

Simpler approach: just load tokenizer + embedding layer and compare.
"""
import argparse
import json
import os
import sys

MODEL_CT = "/models/Step-3.5-Flash-FP8"
MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
PROMPT = "What is 2+2?"


def run_vllm():
    """Load just the embedding weights in vLLM style and compute embedding."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CT, trust_remote_code=False)
    token_ids = tokenizer.encode(PROMPT)

    # Load just the embedding weight from safetensors
    from safetensors.torch import load_file
    import glob

    # Find which shard has embed_tokens
    index = json.load(open(f"{MODEL_CT}/model.safetensors.index.json"))
    embed_file = index["weight_map"]["model.embed_tokens.weight"]
    shard_path = f"{MODEL_CT}/{embed_file}"

    weights = load_file(shard_path)
    embed_weight = weights["model.embed_tokens.weight"]

    # Also get norm weight and lm_head
    norm_file = index["weight_map"]["model.norm.weight"]
    norm_weights = load_file(f"{MODEL_CT}/{norm_file}")
    norm_weight = norm_weights["model.norm.weight"]

    lm_head_file = index["weight_map"]["lm_head.weight"]
    lm_head_weights = load_file(f"{MODEL_CT}/{lm_head_file}")
    lm_head_weight = lm_head_weights["lm_head.weight"]

    # Also load layer 0 input_layernorm
    ln0_file = index["weight_map"]["model.layers.0.input_layernorm.weight"]
    ln0_weights = load_file(f"{MODEL_CT}/{ln0_file}")
    ln0_weight = ln0_weights["model.layers.0.input_layernorm.weight"]

    captured = {
        "engine": "vllm-diag",
        "prompt": PROMPT,
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
    }

    # Compute embedding
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    embedded = torch.nn.functional.embedding(input_ids, embed_weight)

    captured["embed_weight"] = {
        "shape": list(embed_weight.shape),
        "dtype": str(embed_weight.dtype),
        "mean": float(embed_weight.float().mean()),
        "std": float(embed_weight.float().std()),
        "row0_first5": embed_weight[0, :5].float().tolist(),
    }

    captured["embedding"] = {
        "shape": list(embedded.shape),
        "dtype": str(embedded.dtype),
        "mean": float(embedded.float().mean()),
        "std": float(embedded.float().std()),
        "norm": float(embedded.float().norm()),
        "first_token_first5": embedded[0, :5].float().tolist(),
        "last_token_first5": embedded[-1, :5].float().tolist(),
        "first_token_last5": embedded[0, -5:].float().tolist(),
    }

    # Compute GemmaRMSNorm of embedding (first layer input_layernorm)
    # GemmaRMSNorm: x * (1 + w) / sqrt(mean(x^2) + eps)
    x = embedded.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + 1e-5)
    x_normed = x_normed * (1.0 + ln0_weight.float())
    x_normed = x_normed.to(embedded.dtype)

    captured["layer0_input_normed"] = {
        "first_token_first5": x_normed[0, :5].float().tolist(),
        "last_token_first5": x_normed[-1, :5].float().tolist(),
    }

    captured["norm_weight"] = {
        "shape": list(norm_weight.shape),
        "dtype": str(norm_weight.dtype),
        "first5": norm_weight[:5].float().tolist(),
    }

    captured["lm_head_weight"] = {
        "shape": list(lm_head_weight.shape),
        "dtype": str(lm_head_weight.dtype),
        "mean": float(lm_head_weight.float().mean()),
        "std": float(lm_head_weight.float().std()),
        "row0_first5": lm_head_weight[0, :5].float().tolist(),
    }

    captured["ln0_weight"] = {
        "dtype": str(ln0_weight.dtype),
        "first5": ln0_weight[:5].float().tolist(),
    }

    return captured


def run_atom():
    """Same computation in ATOM env."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_HOST, trust_remote_code=False)
    token_ids = tokenizer.encode(PROMPT)

    from safetensors.torch import load_file
    import glob

    index = json.load(open(f"{MODEL_HOST}/model.safetensors.index.json"))
    embed_file = index["weight_map"]["model.embed_tokens.weight"]
    shard_path = f"{MODEL_HOST}/{embed_file}"

    weights = load_file(shard_path)
    embed_weight = weights["model.embed_tokens.weight"]

    norm_file = index["weight_map"]["model.norm.weight"]
    norm_weights = load_file(f"{MODEL_HOST}/{norm_file}")
    norm_weight = norm_weights["model.norm.weight"]

    lm_head_file = index["weight_map"]["lm_head.weight"]
    lm_head_weights = load_file(f"{MODEL_HOST}/{lm_head_file}")
    lm_head_weight = lm_head_weights["lm_head.weight"]

    ln0_file = index["weight_map"]["model.layers.0.input_layernorm.weight"]
    ln0_weights = load_file(f"{MODEL_HOST}/{ln0_file}")
    ln0_weight = ln0_weights["model.layers.0.input_layernorm.weight"]

    captured = {
        "engine": "atom-diag",
        "prompt": PROMPT,
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
    }

    input_ids = torch.tensor(token_ids, dtype=torch.long)
    embedded = torch.nn.functional.embedding(input_ids, embed_weight)

    captured["embed_weight"] = {
        "shape": list(embed_weight.shape),
        "dtype": str(embed_weight.dtype),
        "mean": float(embed_weight.float().mean()),
        "std": float(embed_weight.float().std()),
        "row0_first5": embed_weight[0, :5].float().tolist(),
    }

    captured["embedding"] = {
        "shape": list(embedded.shape),
        "dtype": str(embedded.dtype),
        "mean": float(embedded.float().mean()),
        "std": float(embedded.float().std()),
        "norm": float(embedded.float().norm()),
        "first_token_first5": embedded[0, :5].float().tolist(),
        "last_token_first5": embedded[-1, :5].float().tolist(),
        "first_token_last5": embedded[0, -5:].float().tolist(),
    }

    x = embedded.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + 1e-5)
    x_normed = x_normed * (1.0 + ln0_weight.float())
    x_normed = x_normed.to(embedded.dtype)

    captured["layer0_input_normed"] = {
        "first_token_first5": x_normed[0, :5].float().tolist(),
        "last_token_first5": x_normed[-1, :5].float().tolist(),
    }

    captured["norm_weight"] = {
        "shape": list(norm_weight.shape),
        "dtype": str(norm_weight.dtype),
        "first5": norm_weight[:5].float().tolist(),
    }

    captured["lm_head_weight"] = {
        "shape": list(lm_head_weight.shape),
        "dtype": str(lm_head_weight.dtype),
        "mean": float(lm_head_weight.float().mean()),
        "std": float(lm_head_weight.float().std()),
        "row0_first5": lm_head_weight[0, :5].float().tolist(),
    }

    captured["ln0_weight"] = {
        "dtype": str(ln0_weight.dtype),
        "first5": ln0_weight[:5].float().tolist(),
    }

    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["atom", "vllm"])
    args = parser.parse_args()

    if args.engine == "vllm":
        result = run_vllm()
    else:
        result = run_atom()

    print("__DIAG_JSON_BEGIN__")
    print(json.dumps(result, indent=2))
    print("__DIAG_JSON_END__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
