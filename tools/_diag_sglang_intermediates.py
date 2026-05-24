"""SGLang Step-3.5-Flash intermediate capture.

Runs inside rocm/sgl-dev container with SGLang installed. Patches
sglang.srt.models.step3p5 the same way we patch ATOM's step3p5 module
and dumps a JSON with per-layer captures.
"""

import json
import os
import re
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
# Match the engine-comparison invocation from prior runs.
os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch
import sglang.srt.models.step3p5 as s3
from sglang import Engine

PROMPT = "What is 2+2?"
MODEL = "/models/Step-3.5-Flash-FP8"
OUT_PATH = os.environ.get("SGLANG_DIAG_OUT", "/output/sglang_diag.json")
DATA = {"engine": "sglang", "prompt": PROMPT, "model": MODEL, "intermediates": {}}
INT = DATA["intermediates"]
_current_layer = [-1]


def _layer_idx(module):
    if hasattr(module, "layer_idx"):
        try:
            return int(module.layer_idx)
        except Exception:
            pass
    if hasattr(module, "layer_id"):
        try:
            return int(module.layer_id)
        except Exception:
            pass
    prefix = getattr(module, "prefix", "")
    match = re.search(r"layers\.(\d+)", prefix)
    if match:
        return int(match.group(1))
    return _current_layer[0]


def _summ(value):
    v = value.detach()
    return {
        "shape": list(v.shape),
        "dtype": str(v.dtype),
        "mean": float(v.float().mean().item()),
        "std": float(v.float().std().item()) if v.numel() > 1 else 0.0,
        "norm": float(v.float().norm().item()),
        "abs_max": float(v.float().abs().max().item()),
        "last_tok_first5": v[-1, :5].float().cpu().tolist() if v.dim() >= 2 else v.float().cpu().tolist()[:5],
        "last_tok_norm": float(v[-1].float().norm().item()) if v.dim() >= 2 else float(v.float().norm().item()),
    }


def _capture(store, name, value):
    try:
        store[name] = _summ(value)
    except Exception as exc:
        store[name] = {"error": repr(exc)}


def _capture_topk(store, name, weights, ids):
    try:
        w = weights.detach()
        i = ids.detach()
        store[name] = {
            "weights_shape": list(w.shape),
            "ids_shape": list(i.shape),
            "last_tok_weights": w[-1].float().cpu().tolist(),
            "last_tok_ids": i[-1].int().cpu().tolist(),
        }
    except Exception as exc:
        store[name] = {"error": repr(exc)}


_orig_decoder = s3.Step3p5DecoderLayer.forward
_orig_attn = s3.Step3p5Attention.forward
_orig_moe = s3.Step3p5MoEMLP.forward
_orig_logits = None  # set below if available


def _patched_attn(self, positions, hidden_states, forward_batch):
    idx = _layer_idx(self)
    layer = INT.setdefault(f"layer_{idx}", {})
    _capture(layer, "attn_input", hidden_states)
    try:
        qkv, _ = self.qkv_proj(hidden_states)
        # SGLang uses different sizing accessors; just record the proj output.
        _capture(layer, "qkv_proj_output", qkv)
    except Exception as exc:
        layer["qkv_capture_error"] = repr(exc)
    out = _orig_attn(self, positions, hidden_states, forward_batch)
    _capture(layer, "attn_output", out)
    return out


def _patched_moe(self, hidden_states, forward_batch=None, should_allreduce_fusion=False, use_reduce_scatter=False):
    idx = _layer_idx(self)
    layer = INT.setdefault(f"layer_{int(idx)}", {})
    try:
        if getattr(self, "need_fp32_gate", False):
            router_logits = torch.matmul(
                hidden_states.to(torch.float32), self.gate.weight.t().to(torch.float32)
            )
        else:
            router_logits, _ = self.gate(hidden_states)
        _capture(layer, "router_logits", router_logits)
        topk_output = self.topk(hidden_states, router_logits)
        weights = getattr(topk_output, "topk_weights", None)
        ids = getattr(topk_output, "topk_ids", None)
        if weights is None and isinstance(topk_output, tuple):
            weights, ids = topk_output[:2]
        if weights is not None:
            _capture_topk(layer, "router_topk", weights, ids)
        layer["routed_scaling_factor"] = float(getattr(self, "routed_scaling_factor", 1.0))
    except Exception as exc:
        layer["router_capture_error"] = repr(exc)
    out = _orig_moe(self, hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter)
    _capture(layer, "moe_post_scale", out)
    return out


def _patched_decoder(self, positions, hidden_states, forward_batch, residual, post_residual_addition=None):
    idx = int(getattr(self, "layer_id", getattr(self, "layer_idx", -1)))
    _current_layer[0] = idx
    try:
        layer = INT.setdefault(f"layer_{idx}", {})
        _capture(layer, "layer_input", hidden_states)
        if residual is not None:
            _capture(layer, "layer_residual_in", residual)
        out = _orig_decoder(self, positions, hidden_states, forward_batch, residual, post_residual_addition)
        if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
            _capture(layer, "layer_output", out[0])
        return out
    finally:
        _current_layer[0] = -1


s3.Step3p5Attention.forward = _patched_attn
s3.Step3p5DecoderLayer.forward = _patched_decoder
s3.Step3p5MoEMLP.forward = _patched_moe


def main():
    engine = Engine(
        model_path=MODEL,
        tp_size=1,
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3",
        mem_fraction_static=0.85,
        attention_backend="triton",
        disable_cuda_graph=True,
        log_level="warning",
        max_total_tokens=16384,
    )
    out = engine.generate(
        [PROMPT],
        sampling_params={"temperature": 0.0, "max_new_tokens": 1, "top_p": 1.0},
    )
    DATA["output_text"] = out[0].get("text", "") if out else ""
    engine.shutdown()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as h:
        json.dump(DATA, h, indent=2)
    print("__DIAG_JSON_BEGIN__")
    print(json.dumps({
        "engine": "sglang",
        "prompt": PROMPT,
        "output_text": DATA["output_text"],
        "layers": sorted(INT.keys())[:8],
        "out_path": OUT_PATH,
    }, indent=2))
    print("__DIAG_JSON_END__")


if __name__ == "__main__":
    main()
