"""Diff ATOM and SGLang Step-3.5-Flash per-layer captures.

Reads /tmp/atom_diag_step3p5_merged.json and
/tmp/sglang_diag_step3p5_merged.json (or per-PID dumps) and prints
per-layer divergence between the two engines for the prefill bucket.

For each layer we compare a chosen tensor's `last_tok_first5`,
`last_tok_norm`, and `norm`. We also compare router_topk ids/weights
when both sides have them.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


ATOM_PATH = "/tmp/atom_diag_step3p5_merged.json"
SGLANG_PATH = "/tmp/sglang_diag_step3p5_merged.json"
SGLANG_PER_PID_GLOB = "/tmp/sglang_diag_out/sglang_diag.*.json"


def _pick_atom_capture(merged: dict) -> dict:
    workers = merged.get("workers", {})
    # Pick the worker with the most layers in either bucket.
    best_pid, best_data, best_count = None, None, -1
    for pid, data in workers.items():
        if not isinstance(data, dict):
            continue
        for bucket in ("prefill", "decode"):
            phase = data.get(bucket, {})
            count = sum(1 for k in phase if k.startswith("layer_"))
            if count > best_count:
                best_count = count
                best_pid = pid
                best_data = data
    return best_data or {}


def _pick_sglang_capture() -> dict:
    # Per-PID dumps are richer; aggregate the largest.
    best_data, best_count = None, -1
    for path in glob.glob(SGLANG_PER_PID_GLOB):
        if path.endswith("summary.json"):
            continue
        try:
            data = json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for bucket in ("prefill", "decode"):
            phase = data.get(bucket, {})
            count = sum(1 for k in phase if k.startswith("layer_"))
            if count > best_count:
                best_count = count
                best_data = data
    return best_data or {}


def _safe_get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def _rel_diff(a, b):
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return 0.0
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _vec_cos(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def _compare_tensor(name, atom_t, sgl_t):
    a_norm = _safe_get(atom_t, "last_tok_norm")
    s_norm = _safe_get(sgl_t, "last_tok_norm")
    a_v = _safe_get(atom_t, "last_tok_first5")
    s_v = _safe_get(sgl_t, "last_tok_first5")
    a_global = _safe_get(atom_t, "norm")
    s_global = _safe_get(sgl_t, "norm")
    cos = _vec_cos(a_v, s_v)
    return {
        "atom_norm": a_global,
        "sgl_norm": s_global,
        "norm_rel": _rel_diff(a_global, s_global),
        "atom_last5": a_v,
        "sgl_last5": s_v,
        "last5_cos": cos,
        "atom_last_tok_norm": a_norm,
        "sgl_last_tok_norm": s_norm,
        "last_tok_norm_rel": _rel_diff(a_norm, s_norm),
    }


def _fmt_row(layer_idx, name, cmp):
    cos = cmp["last5_cos"]
    cos_s = f"{cos:+.4f}" if cos is not None else "  n/a "
    nr = cmp["norm_rel"]
    nr_s = f"{nr:.3e}" if nr is not None else "  n/a "
    ar = cmp["atom_norm"]
    sr = cmp["sgl_norm"]
    ar_s = f"{ar:.4g}" if ar is not None else "n/a"
    sr_s = f"{sr:.4g}" if sr is not None else "n/a"
    return f"  L{layer_idx:>2} {name:<24} cos={cos_s} norm_rel={nr_s}  atom={ar_s:>10} sgl={sr_s:>10}"


def _compare_router(layer_idx, atom_l, sgl_l):
    a = atom_l.get("router_topk")
    s = sgl_l.get("router_topk")
    if not isinstance(a, dict) or not isinstance(s, dict):
        return None
    a_ids = a.get("last_tok_ids", [])
    s_ids = s.get("last_tok_ids", [])
    a_w = a.get("last_tok_weights", [])
    s_w = s.get("last_tok_weights", [])
    common_ids = set(a_ids) & set(s_ids)
    return {
        "atom_ids": a_ids,
        "sgl_ids": s_ids,
        "n_common": len(common_ids),
        "atom_w": [round(x, 4) for x in a_w],
        "sgl_w": [round(x, 4) for x in s_w],
    }


def _compare_phase(atom, sgl, bucket):
    a_phase = atom.get(bucket, {})
    s_phase = sgl.get(bucket, {})
    if not a_phase:
        print(f"== {bucket}: ATOM has no {bucket} bucket ==")
        return
    if not s_phase:
        print(f"== {bucket}: SGLang has no {bucket} bucket ==")
        return
    print(f"\n=== {bucket.upper()} bucket ===")
    layer_indices = sorted(
        {int(k.split("_")[1]) for k in (set(a_phase) | set(s_phase)) if k.startswith("layer_")}
    )

    # Compare logits if available
    a_log = a_phase.get("logits")
    s_log = s_phase.get("logits")
    if isinstance(a_log, dict) and isinstance(s_log, dict):
        print(f"  LOGITS  atom_argmax={a_log.get('argmax_last_tok')} "
              f"sgl_argmax={s_log.get('argmax_last_tok')}")
        print(f"          atom_top5={a_log.get('last_tok_top5_ids')}")
        print(f"          sgl_top5 ={s_log.get('last_tok_top5_ids')}")

    # The interesting per-layer tensors that exist in both engines.
    common_keys = ["layer_input", "attn_input", "attn_output", "router_logits"]
    print(f"\n  layer  | first divergence in tensor norm/cosine")
    for idx in layer_indices:
        a_l = a_phase.get(f"layer_{idx}", {})
        s_l = s_phase.get(f"layer_{idx}", {})
        if not a_l or not s_l:
            continue
        printed = False
        for key in common_keys:
            if key not in a_l or key not in s_l:
                continue
            cmp = _compare_tensor(key, a_l[key], s_l[key])
            cos = cmp["last5_cos"]
            nr = cmp["norm_rel"]
            # Only print if interesting (low cos or high rel diff) OR the first layer
            interesting = (
                (cos is not None and cos < 0.999) or
                (nr is not None and nr > 1e-3) or
                idx in (0, 1, 2, 3, 44)
            )
            if interesting:
                print(_fmt_row(idx, key, cmp))
                printed = True
        # router topk for MoE layers
        rt = _compare_router(idx, a_l, s_l)
        if rt is not None:
            print(f"     L{idx:>2} router_topk: common_ids={rt['n_common']}/8 "
                  f"atom={rt['atom_ids']} sgl={rt['sgl_ids']}")
            if rt['n_common'] < 8:
                printed = True
        if printed:
            print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default=ATOM_PATH)
    p.add_argument("--sglang", default=None,
                   help="Merged SGLang JSON path; if absent, picks largest per-PID dump.")
    args = p.parse_args()

    atom = json.loads(Path(args.atom).read_text())
    atom_w = _pick_atom_capture(atom)

    if args.sglang:
        sgl_full = json.loads(Path(args.sglang).read_text())
        # The sglang merged dump uses workers/<pid> too; pick richest.
        if "workers" in sgl_full:
            sgl_w = _pick_atom_capture(sgl_full)
        else:
            sgl_w = sgl_full
    else:
        sgl_w = _pick_sglang_capture()

    print(f"ATOM output : {atom.get('output_text')!r}")
    print(f"SGLang capture source: per-PID dumps in {SGLANG_PER_PID_GLOB}")
    a_pre = sum(1 for k in atom_w.get("prefill", {}) if k.startswith("layer_"))
    s_pre = sum(1 for k in sgl_w.get("prefill", {}) if k.startswith("layer_"))
    a_dec = sum(1 for k in atom_w.get("decode", {}) if k.startswith("layer_"))
    s_dec = sum(1 for k in sgl_w.get("decode", {}) if k.startswith("layer_"))
    print(f"ATOM   layers: prefill={a_pre} decode={a_dec}")
    print(f"SGLang layers: prefill={s_pre} decode={s_dec}")

    _compare_phase(atom_w, sgl_w, "prefill")
    _compare_phase(atom_w, sgl_w, "decode")


if __name__ == "__main__":
    sys.exit(main() or 0)
