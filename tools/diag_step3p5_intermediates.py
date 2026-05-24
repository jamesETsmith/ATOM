#!/usr/bin/env python3
"""Capture Step3p5 intermediates from ATOM and vLLM."""

import argparse
import json
import os
import subprocess
import sys

MODEL_CT = "/models/Step-3.5-Flash-FP8"
MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"
PROMPT = "What is 2+2?"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_IMAGE = "rocm/vllm:rocm7.12.0_gfx94X-dcgpu_ubuntu24.04_py3.12_pytorch_2.9.1_vllm_0.16.0"


def _extract_json(text: str) -> dict:
    begin = "__DIAG_JSON_BEGIN__"
    end = "__DIAG_JSON_END__"
    start = text.find(begin)
    if start == -1:
        raise RuntimeError("missing begin marker")
    start += len(begin)
    finish = text.find(end, start)
    if finish == -1:
        raise RuntimeError("missing end marker")
    return json.loads(text[start:finish].strip())


def _atom_inner_script() -> str:
    return f'''import json
import os
import re

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

import torch
from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
import atom.models.step3p5 as s3

PROMPT = {PROMPT!r}
MODEL = {MODEL_HOST!r}
DATA = {{}}


def _layer_idx(module):
    if hasattr(module, "layer_idx"):
        return int(module.layer_idx)
    prefix = getattr(module, "prefix", "")
    match = re.search(r"layers\\.(\\d+)", prefix)
    return int(match.group(1)) if match else -1


def _capture_tensor(store, name, value):
    value = value.detach()
    store[name] = {{
        "shape": list(value.shape),
        "mean": float(value.float().mean()),
        "std": float(value.float().std()),
        "norm": float(value.float().norm()),
        "abs_max": float(value.float().abs().max()),
        "last_tok_first5": value[-1, :5].float().cpu().tolist(),
        "last_tok_norm": float(value[-1].float().norm()),
    }}


def _capture_topk(store, name, weights, ids):
    weights = weights.detach()
    ids = ids.detach()
    store[name] = {{
        "weights_shape": list(weights.shape),
        "ids_shape": list(ids.shape),
        "last_tok_weights": weights[-1].float().cpu().tolist(),
        "last_tok_ids": ids[-1].int().cpu().tolist(),
    }}


_orig_attn = s3.Step3p5Attention.forward
_orig_decoder = s3.Step3p5DecoderLayer.forward
_orig_model = s3.Step3p5Model.forward
_orig_logits = s3.Step3p5ForCausalLM.compute_logits
_orig_moe = s3.Step3p5MoE.forward


def _patched_attn(self, positions, hidden_states):
    idx = _layer_idx(self)
    layer = DATA.setdefault(f"layer_{{idx}}", {{}})
    _capture_tensor(layer, "attn_input", hidden_states)
    qkv = self.qkv_proj(hidden_states)
    q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
    _capture_tensor(layer, "q_proj", q)
    _capture_tensor(layer, "k_proj", k)
    _capture_tensor(layer, "v_proj", v)
    out = _orig_attn(self, positions, hidden_states)
    _capture_tensor(layer, "attn_output", out)
    return out


def _patched_moe(self, hidden_states):
    layer_idx = _layer_idx(self)
    layer = DATA.setdefault(f"layer_{{int(layer_idx)}}", {{}})
    router_logits = torch.nn.functional.linear(
        hidden_states.float(), self.gate.weight.float()
    )
    _capture_tensor(layer, "router_logits", router_logits)
    routing_weights = torch.sigmoid(router_logits.float())
    scores_for_choice = routing_weights
    if self.router_bias is not None:
        scores_for_choice = scores_for_choice + self.router_bias
    topk_ids = torch.topk(scores_for_choice, self.experts.top_k, dim=-1, sorted=False).indices
    topk_weights = routing_weights.gather(dim=-1, index=topk_ids)
    if self.experts.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    _capture_topk(layer, "router_topk", topk_weights, topk_ids)
    routed = _orig_moe(self, hidden_states)
    _capture_tensor(layer, "moe_routed_raw", routed)
    return routed


def _patched_decoder(self, positions, hidden_states, residual):
    idx = int(self.layer_idx)
    layer = DATA.setdefault(f"layer_{{idx}}", {{}})
    _capture_tensor(layer, "layer_input", hidden_states)
    if residual is not None:
        _capture_tensor(layer, "layer_residual_in", residual)
    if residual is None:
        residual_local = hidden_states
        hidden_states_local = self.input_layernorm(hidden_states)
    else:
        hidden_states_local, residual_local = self.input_layernorm(hidden_states, residual)
    _capture_tensor(layer, "post_input_ln", hidden_states_local)
    _capture_tensor(layer, "residual_after_input_ln", residual_local)
    attn_out = self.self_attn(positions=positions, hidden_states=hidden_states_local)
    _capture_tensor(layer, "post_attention", attn_out)
    post_attn_hidden, post_attn_residual = self.post_attention_layernorm(attn_out, residual_local)
    _capture_tensor(layer, "post_attn_ln", post_attn_hidden)
    _capture_tensor(layer, "residual_after_post_attn_ln", post_attn_residual)
    if self.is_moe:
        routed = self.moe(post_attn_hidden)
        shared = self.share_expert(post_attn_hidden)
        _capture_tensor(layer, "moe_routed", routed)
        _capture_tensor(layer, "moe_shared", shared)
        hidden_out = routed + shared
        _capture_tensor(layer, "moe_sum", hidden_out)
        if self.tp_size > 1:
            hidden_out = s3.tensor_model_parallel_all_reduce(hidden_out)
    else:
        hidden_out = self.mlp(post_attn_hidden)
        _capture_tensor(layer, "mlp_output", hidden_out)
    return hidden_out, post_attn_residual


def _patched_model(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    out = _orig_model(self, input_ids, positions, intermediate_tensors, inputs_embeds)
    if isinstance(out, torch.Tensor):
        _capture_tensor(DATA, "pre_logits_hidden", out)
    return out


def _patched_logits(self, hidden_states):
    _capture_tensor(DATA, "compute_logits_input", hidden_states)
    logits = _orig_logits(self, hidden_states)
    DATA["logits"] = {{
        "shape": list(logits.shape),
        "mean": float(logits.float().mean()),
        "std": float(logits.float().std()),
        "abs_max": float(logits.float().abs().max()),
        "argmax_last_tok": int(logits[-1].argmax().item()),
        "last_tok_top5_ids": logits[-1].topk(5).indices.cpu().tolist(),
        "last_tok_top5_vals": logits[-1].topk(5).values.float().cpu().tolist(),
    }}
    return logits


s3.Step3p5Attention.forward = _patched_attn
s3.Step3p5MoE.forward = _patched_moe
s3.Step3p5DecoderLayer.forward = _patched_decoder
s3.Step3p5Model.forward = _patched_model
s3.Step3p5ForCausalLM.compute_logits = _patched_logits

engine_args = EngineArgs(model=MODEL, tensor_parallel_size=1, enforce_eager=True, kv_cache_dtype="fp8", max_model_len=16384, gpu_memory_utilization=0.92, cudagraph_capture_sizes="[1]", level=0)
llm = engine_args.create_engine()
outputs = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1))
llm.close()

result = {{
    "engine": "atom",
    "prompt": PROMPT,
    "output_text": outputs[0]["text"] if outputs else "",
    "intermediates": DATA,
}}
print("__DIAG_JSON_BEGIN__")
print(json.dumps(result, indent=2))
print("__DIAG_JSON_END__")
'''


def _vllm_inner_script() -> str:
    return f'''import json
import os
import re

os.environ["VLLM_USE_V1"] = "0"

import torch
import vllm.model_executor.models.step3p5 as s3
from vllm import LLM, SamplingParams

PROMPT = {PROMPT!r}
MODEL = {MODEL_CT!r}
DATA = {{}}


def _layer_idx(module):
    if hasattr(module, "layer_idx"):
        return int(module.layer_idx)
    prefix = getattr(module, "prefix", "")
    match = re.search(r"layers\\.(\\d+)", prefix)
    return int(match.group(1)) if match else -1


def _capture_tensor(store, name, value):
    value = value.detach()
    store[name] = {{
        "shape": list(value.shape),
        "mean": float(value.float().mean()),
        "std": float(value.float().std()),
        "norm": float(value.float().norm()),
        "abs_max": float(value.float().abs().max()),
        "last_tok_first5": value[-1, :5].float().cpu().tolist(),
        "last_tok_norm": float(value[-1].float().norm()),
    }}


def _capture_topk(store, name, weights, ids):
    weights = weights.detach()
    ids = ids.detach()
    store[name] = {{
        "weights_shape": list(weights.shape),
        "ids_shape": list(ids.shape),
        "last_tok_weights": weights[-1].float().cpu().tolist(),
        "last_tok_ids": ids[-1].int().cpu().tolist(),
    }}


_orig_decoder = s3.Step3p5DecoderLayer.forward
_orig_logits = s3.Step3p5ForCausalLM.compute_logits
_orig_moe = s3.Step3p5MoE.forward


def _patched_moe(self, hidden_states):
    layer_idx = _layer_idx(self)
    layer = DATA.setdefault(f"layer_{{layer_idx}}", {{}})
    router_logits, _ = self.gate(hidden_states)
    _capture_tensor(layer, "router_logits", router_logits)
    topk_output = self.topk(hidden_states, router_logits)
    if isinstance(topk_output, tuple):
        topk_weights, topk_ids = topk_output
    else:
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
    _capture_topk(layer, "router_topk", topk_weights, topk_ids)
    moe_out = _orig_moe(self, hidden_states)
    _capture_tensor(layer, "moe_routed_raw", moe_out)
    return moe_out


def _patched_decoder(self, positions, hidden_states):
    idx = _layer_idx(self)
    layer = DATA.setdefault(f"layer_{{idx}}", {{}})
    _capture_tensor(layer, "layer_input", hidden_states)
    attn_input = self.input_layernorm(hidden_states)
    _capture_tensor(layer, "post_input_ln", attn_input)
    attn_out = self.self_attn(positions, attn_input)
    _capture_tensor(layer, "post_attention", attn_out)
    post_attn = hidden_states + attn_out
    _capture_tensor(layer, "post_attention_residual", post_attn)
    ffn_input = self.post_attention_layernorm(post_attn)
    _capture_tensor(layer, "post_attn_ln", ffn_input)
    if hasattr(self, "moe") and self.moe is not None:
        moe_out = self.moe(ffn_input)
        _capture_tensor(layer, "moe_sum", moe_out)
        result = post_attn + moe_out
    else:
        mlp_out = self.mlp(ffn_input)
        _capture_tensor(layer, "mlp_output", mlp_out)
        result = post_attn + mlp_out
    _capture_tensor(layer, "layer_output", result)
    return result


def _patched_logits(self, hidden_states):
    _capture_tensor(DATA, "compute_logits_input", hidden_states)
    logits = _orig_logits(self, hidden_states)
    DATA["logits"] = {{
        "shape": list(logits.shape),
        "mean": float(logits.float().mean()),
        "std": float(logits.float().std()),
        "abs_max": float(logits.float().abs().max()),
        "argmax_last_tok": int(logits[-1].argmax().item()),
        "last_tok_top5_ids": logits[-1].topk(5).indices.cpu().tolist(),
        "last_tok_top5_vals": logits[-1].topk(5).values.float().cpu().tolist(),
    }}
    return logits


s3.Step3p5MoE.forward = _patched_moe
s3.Step3p5DecoderLayer.forward = _patched_decoder
s3.Step3p5ForCausalLM.compute_logits = _patched_logits

llm = LLM(model=MODEL, tensor_parallel_size=1, trust_remote_code=False, gpu_memory_utilization=0.92, kv_cache_dtype="fp8", max_model_len=16384, enforce_eager=True, disable_custom_all_reduce=True)
out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=10), use_tqdm=False)[0]
result = {{
    "engine": "vllm",
    "prompt": PROMPT,
    "token_ids": list(out.prompt_token_ids),
    "generated_token": out.outputs[0].token_ids[0] if out.outputs else -1,
    "intermediates": DATA,
}}
print("__DIAG_JSON_BEGIN__")
print(json.dumps(result, indent=2))
print("__DIAG_JSON_END__")
'''


def run_atom(output_path: str | None) -> dict:
    script_path = os.path.join(REPO, "tools", "_diag_atom_intermediates.py")
    with open(script_path, "w") as handle:
        handle.write(_atom_inner_script())
    proc = subprocess.run([sys.executable, script_path], cwd=REPO, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    result = _extract_json(proc.stdout)
    if output_path:
        with open(output_path, "w") as handle:
            json.dump(result, handle, indent=2)
    return result


def run_vllm(output_path: str | None) -> dict:
    script_path = os.path.join(REPO, "tools", "_diag_vllm_intermediates.py")
    with open(script_path, "w") as handle:
        handle.write(_vllm_inner_script())
    command = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", "16g",
        "-e", "HIP_VISIBLE_DEVICES=0",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "VLLM_USE_V1=0",
        "-e", "VLLM_ROCM_USE_AITER_MOE=0",
        "-v", f"{MODEL_HOST}:{MODEL_CT}:ro",
        "-v", f"{REPO}:/workspace/ATOM",
        VLLM_IMAGE,
        "python3", "/workspace/ATOM/tools/_diag_vllm_intermediates.py",
    ]
    proc = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    result = _extract_json(proc.stdout)
    if output_path:
        with open(output_path, "w") as handle:
            json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["atom", "vllm"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run_atom(args.output) if args.engine == "atom" else run_vllm(args.output)
    print(json.dumps({"engine": result["engine"], "keys": sorted(result["intermediates"].keys())[:10]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
