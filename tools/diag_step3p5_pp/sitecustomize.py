"""sitecustomize.py for Step-3.5-Flash intermediate capture.

Placed on PYTHONPATH so every spawned ATOM worker imports it automatically
at interpreter startup. Activates only when ATOM_DIAG_STEP3P5=1 is set.

Strategy:
- Install a meta_path import-hook wrapper that monkey-patches
  atom.models.step3p5 the first time it is imported in this process.
- Capture per-layer intermediates into a process-local dict.
- On atexit, dump JSON to ATOM_DIAG_STEP3P5_OUT.<pid>.json so each worker
  writes its own file. The orchestrator merges them after the run.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys


def _enabled() -> bool:
    return os.environ.get("ATOM_DIAG_STEP3P5", "") == "1"


_DEBUG = os.environ.get("ATOM_DIAG_STEP3P5_DEBUG", "") == "1"
if _DEBUG:
    sys.stderr.write(
        f"[diag] sitecustomize loaded pid={os.getpid()} enabled={_enabled()}\n"
    )
    sys.stderr.flush()

if not _enabled():
    pass
else:
    DATA: dict = {}
    OUT_BASE = os.environ.get("ATOM_DIAG_STEP3P5_OUT", "/tmp/atom_diag_step3p5")

    def _layer_idx(module) -> int:
        if hasattr(module, "layer_idx"):
            try:
                return int(module.layer_idx)
            except Exception:
                pass
        prefix = getattr(module, "prefix", "")
        match = re.search(r"layers\.(\d+)", prefix)
        if match:
            return int(match.group(1))
        # Fall back to thread-local layer index set by the patched decoder.
        return _current_layer_idx[0]

    _current_layer_idx = [-1]  # mutable container, single producer/consumer per worker

    def _summarize_tensor(value):
        import torch  # noqa: F401

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

    def _capture_tensor(store, name, value):
        try:
            store[name] = _summarize_tensor(value)
        except Exception as exc:
            store[name] = {"error": f"capture failed: {exc!r}"}

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
            store[name] = {"error": f"capture failed: {exc!r}"}

    def _install_patches(s3):
        if getattr(s3, "_atom_diag_installed", False):
            return
        s3._atom_diag_installed = True

        import torch  # noqa: F401

        _orig_attn = s3.Step3p5Attention.forward
        _orig_decoder = s3.Step3p5DecoderLayer.forward
        _orig_logits = s3.Step3p5ForCausalLM.compute_logits
        _orig_moe = s3.Step3p5MoE.forward
        _orig_model_fwd = s3.Step3p5Model.forward

        def _bucket_for(hidden_states):
            try:
                seq_len = int(hidden_states.shape[0])
            except Exception:
                return "unknown"
            return "decode" if seq_len == 1 else "prefill"

        def _layer_store(idx, hidden_states):
            bucket = _bucket_for(hidden_states)
            phase = DATA.setdefault(bucket, {})
            return phase.setdefault(f"layer_{idx}", {})

        def _eager_dump():
            path = f"{OUT_BASE}.{os.getpid()}.json"
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as h:
                    json.dump(DATA, h, indent=2)
            except Exception:
                pass

        def _patched_attn(self, positions, hidden_states):
            idx = _layer_idx(self)
            layer = _layer_store(idx, hidden_states)
            _capture_tensor(layer, "attn_input", hidden_states)
            try:
                qkv = self.qkv_proj(hidden_states)
                if isinstance(qkv, tuple):
                    qkv = qkv[0]
                q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
                _capture_tensor(layer, "q_proj", q)
                _capture_tensor(layer, "k_proj", k)
                _capture_tensor(layer, "v_proj", v)
            except Exception as exc:
                layer["qkv_capture_error"] = repr(exc)
            out = _orig_attn(self, positions, hidden_states)
            _capture_tensor(layer, "attn_output", out)
            return out

        def _patched_moe(self, hidden_states):
            idx = _layer_idx(self)
            layer = _layer_store(int(idx), hidden_states)
            try:
                gate_w = self.gate.weight if hasattr(self.gate, "weight") else None
                if gate_w is not None:
                    router_logits = torch.nn.functional.linear(
                        hidden_states.float(), gate_w.float()
                    )
                    _capture_tensor(layer, "router_logits", router_logits)
                    routing_weights = torch.sigmoid(router_logits.float())
                    scores_for_choice = routing_weights
                    if getattr(self, "router_bias", None) is not None:
                        scores_for_choice = scores_for_choice + self.router_bias
                    top_k = self.experts.top_k
                    topk_ids = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False).indices
                    topk_weights = routing_weights.gather(dim=-1, index=topk_ids)
                    if getattr(self.experts, "renormalize", False):
                        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
                    _capture_topk(layer, "router_topk", topk_weights, topk_ids)
            except Exception as exc:
                layer["router_capture_error"] = repr(exc)

            # Replay the body of Step3p5MoE.forward so we can observe the
            # output before the routed_scaling_factor multiply.
            try:
                rl = torch.nn.functional.linear(
                    hidden_states.float(), self.gate.weight.float()
                )
                pre_scale = self.experts(
                    hidden_states=hidden_states, router_logits=rl
                )
                _capture_tensor(layer, "moe_pre_scale", pre_scale)
                layer["routed_scaling_factor"] = float(self.routed_scaling_factor)
                if self.routed_scaling_factor != 1.0:
                    routed = pre_scale * self.routed_scaling_factor
                else:
                    routed = pre_scale
                _capture_tensor(layer, "moe_post_scale", routed)
                return routed
            except Exception as exc:
                # Fall back to original path if our replay fails.
                layer["replay_error"] = repr(exc)
                routed = _orig_moe(self, hidden_states)
                _capture_tensor(layer, "moe_post_scale", routed)
                return routed

        def _patched_decoder(self, positions, hidden_states, residual):
            idx = int(self.layer_idx)
            _current_layer_idx[0] = idx
            try:
                layer = _layer_store(idx, hidden_states)
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
                _eager_dump()
                return hidden_out, post_attn_residual
            finally:
                _current_layer_idx[0] = -1

        def _patched_logits(self, hidden_states):
            bucket = _bucket_for(hidden_states)
            phase = DATA.setdefault(bucket, {})
            _capture_tensor(phase, "compute_logits_input", hidden_states)
            logits = _orig_logits(self, hidden_states)
            try:
                last = logits[-1]
                top5 = last.topk(5)
                phase["logits"] = {
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                    "mean": float(logits.float().mean().item()),
                    "std": float(logits.float().std().item()) if logits.numel() > 1 else 0.0,
                    "abs_max": float(logits.float().abs().max().item()),
                    "argmax_last_tok": int(last.argmax().item()),
                    "last_tok_top5_ids": top5.indices.cpu().tolist(),
                    "last_tok_top5_vals": top5.values.float().cpu().tolist(),
                }
            except Exception as exc:
                phase["logits"] = {"error": repr(exc)}
            _eager_dump()
            return logits

        def _patched_model_fwd(self, input_ids, positions,
                               intermediate_tensors=None, inputs_embeds=None):
            # Wrap self.norm (just once per instance) to capture the value
            # that enters the final RMSNorm and the value that leaves it.
            if not getattr(self, "_atom_diag_norm_wrapped", False):
                _orig_norm_fwd = self.norm.forward

                def _wrapped_norm(hidden_states, residual=None, *args, **kwargs):
                    try:
                        bucket = _bucket_for(hidden_states)
                        store = DATA.setdefault(bucket, {}).setdefault("final", {})
                        _capture_tensor(store, "final_hidden_in", hidden_states)
                        if residual is not None:
                            _capture_tensor(store, "final_residual_in", residual)
                            pre_norm = hidden_states + residual
                            _capture_tensor(store, "pre_norm_hidden", pre_norm)
                        else:
                            _capture_tensor(store, "pre_norm_hidden", hidden_states)
                    except Exception as exc:
                        DATA.setdefault("__diag_meta__", {})["norm_in_capture_error"] = repr(exc)
                    out = _orig_norm_fwd(hidden_states, residual, *args, **kwargs)
                    try:
                        normed = out[0] if isinstance(out, tuple) else out
                        bucket = _bucket_for(normed)
                        store = DATA.setdefault(bucket, {}).setdefault("final", {})
                        _capture_tensor(store, "post_norm_hidden", normed)
                        _eager_dump()
                    except Exception as exc:
                        DATA.setdefault("__diag_meta__", {})["norm_out_capture_error"] = repr(exc)
                    return out

                self.norm.forward = _wrapped_norm
                self._atom_diag_norm_wrapped = True
            return _orig_model_fwd(self, input_ids, positions, intermediate_tensors, inputs_embeds)

        s3.Step3p5Attention.forward = _patched_attn
        s3.Step3p5MoE.forward = _patched_moe
        s3.Step3p5DecoderLayer.forward = _patched_decoder
        s3.Step3p5ForCausalLM.compute_logits = _patched_logits
        s3.Step3p5Model.forward = _patched_model_fwd

        # Stamp on the module so we can confirm activation in the dump.
        DATA["__diag_meta__"] = {
            "pid": os.getpid(),
            "module_path": getattr(s3, "__file__", "?"),
            "patches_installed": True,
        }

    # ----- Import hook -----
    # Install a meta_path finder that defers to the real machinery for
    # atom.models.step3p5 and patches the resulting module before it returns
    # to the importer. This avoids the audit-hook ordering issue where the
    # module is not yet in sys.modules when the "import" event fires.
    import importlib.abc
    import importlib.machinery
    import importlib.util

    TARGET = "atom.models.step3p5"

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, real_loader):
            self._real_loader = real_loader

        def create_module(self, spec):
            if hasattr(self._real_loader, "create_module"):
                return self._real_loader.create_module(spec)
            return None

        def exec_module(self, module):
            self._real_loader.exec_module(module)
            try:
                _install_patches(module)
                if _DEBUG:
                    sys.stderr.write(
                        f"[diag] patched {TARGET} pid={os.getpid()}\n"
                    )
                    sys.stderr.flush()
            except Exception as exc:
                sys.stderr.write(
                    f"[diag] patch failed pid={os.getpid()}: {exc!r}\n"
                )
                sys.stderr.flush()

    class _PatchingFinder(importlib.abc.MetaPathFinder):
        _busy = False

        def find_spec(self, name, path=None, target=None):
            if name != TARGET or _PatchingFinder._busy:
                return None
            _PatchingFinder._busy = True
            try:
                # Re-resolve through the rest of sys.meta_path.
                spec = None
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    if hasattr(finder, "find_spec"):
                        spec = finder.find_spec(name, path, target)
                        if spec is not None:
                            break
                if spec is None or spec.loader is None:
                    return None
                spec.loader = _PatchingLoader(spec.loader)
                return spec
            finally:
                _PatchingFinder._busy = False

    sys.meta_path.insert(0, _PatchingFinder())

    # Handle case where module was already imported before sitecustomize ran.
    if TARGET in sys.modules:
        try:
            _install_patches(sys.modules[TARGET])
            if _DEBUG:
                sys.stderr.write(
                    f"[diag] patched-existing {TARGET} pid={os.getpid()}\n"
                )
                sys.stderr.flush()
        except Exception as exc:
            DATA.setdefault("__diag_meta__", {})["patch_error_initial"] = repr(exc)

    def _dump_atexit():
        if not DATA:
            return
        path = f"{OUT_BASE}.{os.getpid()}.json"
        try:
            with open(path, "w") as handle:
                json.dump(DATA, handle, indent=2)
        except Exception as exc:
            sys.stderr.write(f"[diag] failed to write {path}: {exc!r}\n")

    atexit.register(_dump_atexit)
