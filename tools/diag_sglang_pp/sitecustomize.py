"""sitecustomize for SGLang Step-3.5-Flash intermediate capture.

Mounted into a rocm/sgl-dev container on PYTHONPATH so every worker that
imports `sglang.srt.models.step3p5` gets monkey-patched at module-load
time. Captured intermediates are written to one JSON per worker PID at
`$SGLANG_DIAG_OUT.<pid>.json`.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys


_DEBUG = os.environ.get("SGLANG_DIAG_DEBUG", "") == "1"
_OUT_BASE = os.environ.get("SGLANG_DIAG_OUT", "/output/sglang_diag")


def _enabled() -> bool:
    return os.environ.get("SGLANG_DIAG", "1") == "1"


if _DEBUG:
    sys.stderr.write(f"[sgldiag] sitecustomize loaded pid={os.getpid()} enabled={_enabled()}\n")
    sys.stderr.flush()


if _enabled():
    DATA: dict = {}
    _current_layer = [-1]

    def _layer_idx(module):
        for attr in ("layer_id", "layer_idx"):
            if hasattr(module, attr):
                try:
                    return int(getattr(module, attr))
                except Exception:
                    pass
        prefix = getattr(module, "prefix", "")
        match = re.search(r"layers\.(\d+)", prefix)
        if match:
            return int(match.group(1))
        return _current_layer[0]

    def _summ(value):
        import torch

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

    def _eager_dump():
        path = f"{_OUT_BASE}.{os.getpid()}.json"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as h:
                json.dump(DATA, h, indent=2)
        except Exception:
            pass

    def _install_patches(s3):
        if getattr(s3, "_sgldiag_installed", False):
            return
        s3._sgldiag_installed = True

        import torch

        _orig_decoder = s3.Step3p5DecoderLayer.forward
        _orig_attn = s3.Step3p5Attention.forward
        _orig_moe = s3.Step3p5MoEMLP.forward
        _orig_model_fwd = s3.Step3p5Model.forward
        _orig_causal_fwd = s3.Step3p5ForCausalLM.forward

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

        def _patched_attn(self, positions, hidden_states, forward_batch):
            idx = _layer_idx(self)
            layer = _layer_store(idx, hidden_states)
            _capture(layer, "attn_input", hidden_states)
            try:
                qkv, _ = self.qkv_proj(hidden_states)
                _capture(layer, "qkv_proj_output", qkv)
            except Exception as exc:
                layer["qkv_capture_error"] = repr(exc)
            out = _orig_attn(self, positions, hidden_states, forward_batch)
            _capture(layer, "attn_output", out)
            return out

        def _patched_moe(self, hidden_states, forward_batch=None,
                         should_allreduce_fusion=False, use_reduce_scatter=False):
            idx = _layer_idx(self)
            layer = _layer_store(int(idx), hidden_states)
            try:
                if getattr(self, "need_fp32_gate", False):
                    router_logits = torch.matmul(
                        hidden_states.to(torch.float32),
                        self.gate.weight.t().to(torch.float32),
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
                layer["routed_scaling_factor"] = float(
                    getattr(self, "routed_scaling_factor", 1.0)
                )
            except Exception as exc:
                layer["router_capture_error"] = repr(exc)
            try:
                out = _orig_moe(
                    self,
                    hidden_states,
                    forward_batch,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                )
            except TypeError:
                # Some sglang versions take fewer args.
                out = _orig_moe(self, hidden_states)
            _capture(layer, "moe_post_scale", out)
            return out

        def _patched_decoder(self, positions, hidden_states, forward_batch,
                             residual, post_residual_addition=None):
            idx = int(getattr(self, "layer_id", getattr(self, "layer_idx", -1)))
            _current_layer[0] = idx
            try:
                layer = _layer_store(idx, hidden_states)
                _capture(layer, "layer_input", hidden_states)
                if residual is not None:
                    _capture(layer, "layer_residual_in", residual)
                out = _orig_decoder(
                    self, positions, hidden_states, forward_batch,
                    residual, post_residual_addition,
                )
                if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
                    _capture(layer, "layer_output", out[0])
                # Eagerly dump on every forward so we survive hard-kill at shutdown.
                _eager_dump()
                return out
            finally:
                _current_layer[0] = -1

        def _final_store(hidden_states):
            bucket = _bucket_for(hidden_states)
            phase = DATA.setdefault(bucket, {})
            return phase.setdefault("final", {})

        def _patched_model_fwd(self, input_ids, positions, forward_batch,
                               input_embeds=None, pp_proxy_tensors=None):
            out = _orig_model_fwd(
                self, input_ids, positions, forward_batch,
                input_embeds, pp_proxy_tensors,
            )
            try:
                if isinstance(out, tuple) and len(out) >= 2:
                    hidden_states, hs_before_norm = out[0], out[1]
                    if torch.is_tensor(hidden_states):
                        store = _final_store(hidden_states)
                        _capture(store, "post_norm_hidden", hidden_states)
                        if torch.is_tensor(hs_before_norm):
                            _capture(store, "pre_norm_hidden", hs_before_norm)
                        _eager_dump()
            except Exception as exc:
                DATA.setdefault("__diag_meta__", {})["model_fwd_capture_error"] = repr(exc)
            return out

        def _patched_causal_fwd(self, input_ids, positions, forward_batch,
                                input_embeds=None, pp_proxy_tensors=None):
            out = _orig_causal_fwd(
                self, input_ids, positions, forward_batch,
                input_embeds, pp_proxy_tensors,
            )
            try:
                # Bucket by input_ids length, not logits shape: prefill inputs
                # have multiple tokens but produce only [1, vocab] logits.
                try:
                    n_in = int(input_ids.shape[0])
                except Exception:
                    n_in = -1
                bucket = "decode" if n_in == 1 else "prefill"

                logits = None
                for attr in ("next_token_logits", "logits"):
                    if hasattr(out, attr):
                        cand = getattr(out, attr)
                        if torch.is_tensor(cand):
                            logits = cand
                            break
                if logits is None and torch.is_tensor(out):
                    logits = out
                if logits is not None:
                    store = DATA.setdefault(bucket, {}).setdefault("final", {})
                    _capture(store, "logits", logits)
                    last = logits[-1] if logits.dim() >= 2 else logits
                    last_f = last.float()
                    vals, ids = torch.topk(last_f, k=min(5, last_f.numel()))
                    store["last_tok_top5_ids"] = ids.cpu().tolist()
                    store["last_tok_top5_vals"] = vals.cpu().tolist()
                    store["last_tok_argmax"] = int(last_f.argmax().item())
                    store["__causal_input_n_tokens"] = n_in
                    _eager_dump()
            except Exception as exc:
                DATA.setdefault("__diag_meta__", {})["causal_fwd_capture_error"] = repr(exc)
            return out

        s3.Step3p5Attention.forward = _patched_attn
        s3.Step3p5DecoderLayer.forward = _patched_decoder
        s3.Step3p5MoEMLP.forward = _patched_moe
        s3.Step3p5Model.forward = _patched_model_fwd
        s3.Step3p5ForCausalLM.forward = _patched_causal_fwd

        DATA["__diag_meta__"] = {
            "pid": os.getpid(),
            "module_path": getattr(s3, "__file__", "?"),
            "patches_installed": True,
        }

    # Install meta_path finder that defers to default loader, then patches.
    import importlib.abc
    import importlib.machinery
    import importlib.util

    TARGET = "sglang.srt.models.step3p5"

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, real_loader):
            self._real = real_loader

        def create_module(self, spec):
            if hasattr(self._real, "create_module"):
                return self._real.create_module(spec)
            return None

        def exec_module(self, module):
            self._real.exec_module(module)
            try:
                _install_patches(module)
                if _DEBUG:
                    sys.stderr.write(f"[sgldiag] patched {TARGET} pid={os.getpid()}\n")
                    sys.stderr.flush()
            except Exception as exc:
                sys.stderr.write(f"[sgldiag] patch failed pid={os.getpid()}: {exc!r}\n")
                sys.stderr.flush()

    class _PatchingFinder(importlib.abc.MetaPathFinder):
        _busy = False

        def find_spec(self, name, path=None, target=None):
            if name != TARGET or _PatchingFinder._busy:
                return None
            _PatchingFinder._busy = True
            try:
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

    if TARGET in sys.modules:
        try:
            _install_patches(sys.modules[TARGET])
        except Exception as exc:
            DATA.setdefault("__diag_meta__", {})["patch_error_initial"] = repr(exc)

    def _dump_atexit():
        path = f"{_OUT_BASE}.{os.getpid()}.json"
        payload = dict(DATA) if DATA else {}
        payload.setdefault("__diag_meta__", {})
        meta = payload["__diag_meta__"]
        meta["pid"] = os.getpid()
        meta["target_in_sys_modules_at_exit"] = TARGET in sys.modules
        if TARGET in sys.modules:
            mod = sys.modules[TARGET]
            meta["module_file"] = getattr(mod, "__file__", "?")
            meta["patches_installed"] = bool(getattr(mod, "_sgldiag_installed", False))
        meta["captured_layers"] = sorted(
            k for k in payload.keys() if k.startswith("layer_")
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as h:
                json.dump(payload, h, indent=2)
            if _DEBUG:
                sys.stderr.write(
                    f"[sgldiag] dumped {len(meta['captured_layers'])} layers -> {path}\n"
                )
                sys.stderr.flush()
        except Exception as exc:
            sys.stderr.write(f"[sgldiag] failed to write {path}: {exc!r}\n")

    atexit.register(_dump_atexit)
