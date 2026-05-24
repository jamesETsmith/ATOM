#!/usr/bin/env python3
"""Step-3.5-Flash decode path diagnostic.

Hooks into the model to compare hidden state statistics between prefill
and decode steps.  Run with the model server:

    ATOM_DIAG_STEP3P5=1 python -m atom.entrypoints.openai_server \
        --model /data/jamesmit/models/Step-3.5-Flash-FP8 \
        --kv_cache_dtype fp8

Then send a request.  The diagnostic prints per-layer norm/std/range for
every forward pass so you can spot where decode diverges from prefill.
"""

import os
import sys
import logging
import functools

logger = logging.getLogger("step3p5_diag")

_HOOKS_INSTALLED = False


def _stat_str(t, label=""):
    """One-line summary of a tensor."""
    if t is None:
        return f"{label}None"
    if not t.is_floating_point():
        return f"{label}int shape={list(t.shape)}"
    # Detach and float for stats
    x = t.detach().float()
    return (
        f"{label}shape={list(t.shape)} "
        f"dtype={t.dtype} "
        f"mean={x.mean().item():.4f} "
        f"std={x.std().item():.4f} "
        f"min={x.min().item():.4f} "
        f"max={x.max().item():.4f} "
        f"norm={x.norm().item():.4f} "
        f"absmax={x.abs().max().item():.4f}"
    )


def _hook_decoder_layer(module, args, output, layer_idx):
    """Post-forward hook for Step3p5DecoderLayer."""
    hidden_states, residual = output
    n_tokens = hidden_states.shape[0]
    phase = "PREFILL" if n_tokens > 1 else "DECODE"
    logger.info(
        f"[{phase}] Layer {layer_idx:2d} | "
        f"hidden: {_stat_str(hidden_states)} | "
        f"residual: {_stat_str(residual)}"
    )


def _hook_attention(module, args, output, layer_idx):
    """Post-forward hook for Step3p5Attention."""
    # output is the attn_output tensor
    attn_out = output
    n_tokens = attn_out.shape[0]
    phase = "PREFILL" if n_tokens > 1 else "DECODE"
    logger.info(
        f"[{phase}] Layer {layer_idx:2d} attn_out: {_stat_str(attn_out)}"
    )


def _hook_moe(module, args, output, layer_idx):
    """Post-forward hook for Step3p5MoE."""
    n_tokens = output.shape[0]
    phase = "PREFILL" if n_tokens > 1 else "DECODE"
    logger.info(
        f"[{phase}] Layer {layer_idx:2d} moe_out: {_stat_str(output)}"
    )


def _hook_g_proj(module, args, output, layer_idx):
    """Post-forward hook for g_proj to check gate values."""
    n_tokens = output.shape[0]
    phase = "PREFILL" if n_tokens > 1 else "DECODE"
    import torch
    gate = torch.sigmoid(output.detach().float())
    logger.info(
        f"[{phase}] Layer {layer_idx:2d} g_proj | "
        f"raw: {_stat_str(output)} | "
        f"sigmoid: mean={gate.mean().item():.4f} std={gate.std().item():.4f} "
        f"min={gate.min().item():.4f} max={gate.max().item():.4f}"
    )


def install_hooks(model):
    """Install diagnostic hooks on a Step3p5ForCausalLM model."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    logger.info("=== Step3p5 decode diagnostic hooks installed ===")

    # Find the model backbone
    backbone = model
    if hasattr(model, "model"):
        backbone = model.model
    if hasattr(backbone, "model"):
        backbone = backbone.model

    for layer_idx, layer in enumerate(backbone.layers):
        if layer is None:
            continue

        # Hook decoder layer
        layer.register_forward_hook(
            functools.partial(_hook_decoder_layer, layer_idx=layer_idx)
        )

        # Hook attention output
        if hasattr(layer, "self_attn"):
            layer.self_attn.register_forward_hook(
                functools.partial(_hook_attention, layer_idx=layer_idx)
            )

            # Hook g_proj if present
            if hasattr(layer.self_attn, "g_proj") and layer.self_attn.g_proj is not None:
                layer.self_attn.g_proj.register_forward_hook(
                    functools.partial(_hook_g_proj, layer_idx=layer_idx)
                )

        # Hook MoE
        if hasattr(layer, "moe") and layer.moe is not None:
            layer.moe.register_forward_hook(
                functools.partial(_hook_moe, layer_idx=layer_idx)
            )

    logger.info(f"Hooked {len(backbone.layers)} decoder layers")


def maybe_install():
    """Call from model_runner after model construction if env var is set."""
    if os.environ.get("ATOM_DIAG_STEP3P5", "0") == "1":
        return True
    return False
