#!/usr/bin/env python3
"""Minimal Step-3.5-Flash decode diagnostic.

Usage: After starting the server, send requests with temperature=0 and check
the server stderr for diagnostic output.

This script patches Step3p5DecoderLayer.forward and Step3p5MoE.forward
to print hidden state norms. It must be imported BEFORE the model is loaded.

Integrate by adding to atom/models/step3p5.py:

    # At the bottom of the file, after class definitions:
    import os
    if os.environ.get("ATOM_DIAG_STEP3P5") == "1":
        from tools.diag_step3p5_minimal import patch_step3p5
        patch_step3p5()
"""

import torch
import sys

_call_count = 0
_max_log_calls = 100  # Stop logging after this many calls to avoid flooding


def _norm_str(t):
    if t is None:
        return "None"
    x = t.detach().float()
    return f"norm={x.norm():.2f} absmax={x.abs().max():.4f} mean={x.mean():.4f}"


def patch_step3p5():
    """Monkey-patch Step3p5 model classes to add decode diagnostics."""
    from atom.models.step3p5 import Step3p5DecoderLayer, Step3p5MoE, Step3p5Attention

    orig_layer_fwd = Step3p5DecoderLayer.forward
    orig_moe_fwd = Step3p5MoE.forward
    orig_attn_fwd = Step3p5Attention.forward

    def diag_layer_fwd(self, positions, hidden_states, residual):
        global _call_count
        hidden_states_out, residual_out = orig_layer_fwd(self, positions, hidden_states, residual)

        if _call_count < _max_log_calls:
            n = hidden_states_out.shape[0]
            phase = "P" if n > 1 else "D"
            print(
                f"[{phase}] L{self.layer_idx:02d} "
                f"in:{_norm_str(hidden_states)} "
                f"out:{_norm_str(hidden_states_out)} "
                f"res:{_norm_str(residual_out)}",
                file=sys.stderr, flush=True,
            )
            _call_count += 1

        return hidden_states_out, residual_out

    def diag_moe_fwd(self, hidden_states):
        global _call_count
        routed_out = orig_moe_fwd(self, hidden_states)

        if _call_count < _max_log_calls:
            n = hidden_states.shape[0]
            phase = "P" if n > 1 else "D"
            print(
                f"  [{phase}] MoE "
                f"in:{_norm_str(hidden_states)} "
                f"routed:{_norm_str(routed_out)} "
                f"scale={self.routed_scaling_factor}",
                file=sys.stderr, flush=True,
            )

        return routed_out

    def diag_attn_fwd(self, positions, hidden_states):
        global _call_count
        attn_out = orig_attn_fwd(self, positions, hidden_states)

        if _call_count < _max_log_calls:
            n = hidden_states.shape[0]
            phase = "P" if n > 1 else "D"
            # Also check g_proj gate if present
            gate_info = ""
            if self.g_proj is not None:
                with torch.no_grad():
                    gate = torch.sigmoid(self.g_proj(hidden_states).float())
                    gate_info = f" gate:mean={gate.mean():.3f},std={gate.std():.3f}"
            print(
                f"  [{phase}] Attn "
                f"in:{_norm_str(hidden_states)} "
                f"out:{_norm_str(attn_out)}"
                f"{gate_info}",
                file=sys.stderr, flush=True,
            )

        return attn_out

    Step3p5DecoderLayer.forward = diag_layer_fwd
    Step3p5MoE.forward = diag_moe_fwd
    Step3p5Attention.forward = diag_attn_fwd

    print("[DIAG] Step3p5 decode diagnostics enabled", file=sys.stderr, flush=True)
