# SPDX-License-Identifier: Apache-2.0
"""Inference-only Step-3.5-Flash model (stepfun-ai/Step-3.5-Flash).

196B sparse MoE (288 experts, top-8, ~11B active).  Hybrid full/sliding-window
GQA with per-head sigmoid gating, per-layer RoPE theta, and QK norms.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F

from aiter import QuantType
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_config.step3p5 import Step3p5Config
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import get_forward_context
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile
from torch import nn

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Step3p5MLP(nn.Module):
    """SwiGLU MLP with optional activation clamping (swiglu_limits)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        swiglu_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if self.swiglu_limit > 0:
            gate, up = gate_up.chunk(2, dim=-1)
            x = F.silu(gate).clamp(max=self.swiglu_limit) * up.clamp(
                min=-self.swiglu_limit, max=self.swiglu_limit
            )
        else:
            x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def _unshuffle_weight(
    w: torch.Tensor, layout: tuple[int, int] = (16, 16)
) -> torch.Tensor:
    """Reverse AITER shuffle_weight (inverse permutation)."""
    IN, IK = layout
    BK = IK * 2
    K = 16 // w.element_size()
    BN = IN
    x = w.view(-1, w.shape[-2] // BN, w.shape[-1] // BK, BK // K, BN, K)
    x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
    return x.view(*w.shape)


def _dequant_fp8_blockscale(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 block-scaled weight to bf16.

    Args:
        weight_fp8: [N, K] float8_e4m3fnuz weight (already unshuffled).
        scale: [ceil(N/block_n), ceil(K/block_k)] float32 block scales.
    Returns:
        [N, K] bf16 dequantized weight.
    """
    N, K = weight_fp8.shape
    sn, sk = scale.shape
    out = weight_fp8.to(torch.float32).view(sn, block_n, sk, block_k)
    out = out * scale[:, None, :, None]
    return out.reshape(N, K).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Swiglustep MoE custom op — opaque to torch.compile / Dynamo so it can
# use a Python loop over experts without causing graph breaks.
# ---------------------------------------------------------------------------


def _dequant_expert(
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    eid: int,
) -> torch.Tensor:
    """Dequantize a single expert's FP8 weight to bf16 on-the-fly.

    Avoids materialising the full [E, …] bf16 tensor — saves ~8.4 GiB
    per swiglustep MoE layer on TP=1.
    """
    return _dequant_fp8_blockscale(
        _unshuffle_weight(w_fp8[eid]), w_scale[eid]
    )


def _swiglustep_unfused_compute(
    x_bf16: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_fp8: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bf16: torch.Tensor | None,
    w2_bf16: torch.Tensor | None,
    num_experts: int,
    inter_dim: int,
    inter_pad: int,
    hidden_size: int,
    limit: float,
    top_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Core swiglustep computation.

    Two modes depending on whether bf16 buffers are materialised:

    * **Eager (bf16 is None):** dequantizes per-expert on-the-fly.
      Uses ~20 MiB peak (one expert at a time) instead of ~8.4 GiB.
    * **Graph capture (bf16 provided):** uses pre-materialised bf16
      buffers for fast index-gather in fixed-shape chunks.
    """
    M = x_bf16.shape[0]
    if w13_bf16 is not None and w2_bf16 is not None:
        # Pre-materialised path (graph capture): chunked BMM gather.
        _CHUNK = 64
        output = torch.zeros(M, hidden_size, dtype=torch.bfloat16,
                             device=x_bf16.device)
        for k in range(top_k):
            eids = topk_ids[:, k]
            wts_k = topk_weights[:, k:k + 1]
            for c_start in range(0, M, _CHUNK):
                c_end = min(c_start + _CHUNK, M)
                c_eids = eids[c_start:c_end]
                w13_c = w13_bf16[c_eids.long()]
                x_c = x_bf16[c_start:c_end]
                gate_up = torch.bmm(
                    w13_c, x_c.unsqueeze(2)
                ).squeeze(2)
                gate = gate_up[:, :inter_dim]
                up = gate_up[:, inter_pad:inter_pad + inter_dim]
                act = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
                w2_c = w2_bf16[c_eids.long(), :, :inter_dim]
                down = torch.bmm(
                    w2_c, act.unsqueeze(2)
                ).squeeze(2)
                output[c_start:c_end] += wts_k[c_start:c_end] * down
        return output.to(out_dtype)
    else:
        # On-the-fly dequant path (eager / warmup / prefill).
        output = torch.zeros(M, hidden_size, dtype=out_dtype,
                             device=x_bf16.device)
        for eid in range(num_experts):
            mask = (topk_ids == eid)
            token_mask = mask.any(dim=1)
            if not token_mask.any():
                continue
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            x_sel = x_bf16[token_indices]
            w13 = _dequant_expert(w13_fp8, w13_scale, eid)
            gate_up = x_sel @ w13.t()
            gate = gate_up[:, :inter_dim]
            up = gate_up[:, inter_pad:inter_pad + inter_dim]
            act = F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
            w2 = _dequant_expert(w2_fp8, w2_scale, eid)
            w2 = w2[:, :inter_dim]
            down = act @ w2.t()
            wts = topk_weights[token_indices].unsqueeze(-1)
            expert_wts = mask[token_indices].float().unsqueeze(-1)
            per_token_wt = (wts * expert_wts).sum(dim=1)
            output[token_indices] += (per_token_wt * down).to(output.dtype)
        return output


def _swiglustep_moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Unfused MoE forward with swiglustep activation (custom op impl).

    Three execution modes:

    * **EP mode**: uses the FusedMoE modular kernel's ``_prepare`` /
      ``_finalize`` for MoRI dispatch/combine, with unfused swiglustep
      computation on the dispatched local tokens in between.

    * **Graph-capture path** (non-EP): iterates over ``top_k``
      assignments, gathers expert weights for all M tokens per
      assignment, and accumulates via bmm.

    * **Eager path** (non-EP): iterates over experts, selects tokens
      via ``nonzero``, and runs standard matmul.
    """
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]

    if not self._swiglustep_ready:
        self._prepare_swiglustep_weights()

    # Lazily materialise bf16 buffers for CUDA graph capture.
    # This happens AFTER warmup/KV-cache sizing so it does not inflate
    # peak_torch, which is the key to fitting MTP in memory.
    if torch.cuda.is_current_stream_capturing():
        self._materialize_swiglustep_bf16()

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        use_grouped_topk=False,
        top_k=self.top_k,
        renormalize=self.renormalize,
        scoring_func=self.scoring_func,
        e_score_correction_bias=self.router_bias,
    )

    limit = self.swiglu_limit
    inter_dim = self._inter_dim
    inter_pad = self._inter_pad

    if self._use_ep:
        # --- EP mode: dispatch via modular kernel, unfused local compute ---
        fused_experts = self.experts.quant_method.fused_experts
        (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        ) = fused_experts._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            self.num_experts,
            self.experts.expert_mask,
            False,
            QuantType.No,
        )

        context = get_forward_context().context
        num_dispatchers = fused_experts.prepare_finalize.num_dispatchers()
        total_valid = context.graph_bs * self.top_k * num_dispatchers
        if total_valid < dispatch_a1.shape[0] and not context.is_prefill:
            dispatch_a1 = dispatch_a1[:total_valid]
            dispatch_ids = dispatch_ids[:total_valid]
            dispatch_weights = dispatch_weights[:total_valid]

        x_disp = dispatch_a1.to(torch.bfloat16)
        local_E = self._w13_fp8.shape[0]
        disp_ids_2d = dispatch_ids.unsqueeze(1) if dispatch_ids.dim() == 1 else dispatch_ids
        disp_wts_2d = dispatch_weights.unsqueeze(1) if dispatch_weights.dim() == 1 else dispatch_weights

        # Remap global expert IDs → local indices.  MoRI dispatch
        # returns global IDs but local weight tensors are sized
        # [local_E, ...], so we must translate before indexing.
        expert_map = self.experts.expert_map  # [global_E] → local or -1
        if expert_map is not None:
            disp_ids_2d = expert_map[disp_ids_2d.long()].to(disp_ids_2d.dtype)

        fused_out = _swiglustep_unfused_compute(
            x_disp, disp_wts_2d, disp_ids_2d,
            self._w13_fp8, self._w13_scale,
            self._w2_fp8, self._w2_scale,
            self._w13_bf16, self._w2_bf16,
            local_E, inter_dim, inter_pad,
            self.hidden_size, limit, self.top_k,
            hidden_states.dtype,
        )

        return fused_experts._finalize(
            None, fused_out, hidden_states,
            topk_weights, topk_ids, False,
        )
    else:
        # --- Non-EP: direct compute on all experts ---
        x_bf16 = hidden_states.to(torch.bfloat16)
        return _swiglustep_unfused_compute(
            x_bf16, topk_weights, topk_ids,
            self._w13_fp8, self._w13_scale,
            self._w2_fp8, self._w2_scale,
            self._w13_bf16, self._w2_bf16,
            self.num_experts, inter_dim, inter_pad,
            self.hidden_size, limit, self.top_k,
            hidden_states.dtype,
        )


def _swiglustep_moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="swiglustep_moe_forward",
    op_func=_swiglustep_moe_forward,
    mutates_args=[],
    fake_impl=_swiglustep_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class Step3p5MoE(nn.Module):
    """Step-3.5-Flash routed MoE: sigmoid routing with router bias.

    The shared expert lives in ``Step3p5DecoderLayer`` (not here) so that
    its parameter path (``layers.N.share_expert.*``) matches the HF
    checkpoint layout.
    """

    def __init__(
        self,
        config: Step3p5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
    ):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.swiglu_limit = config.get_swiglu_limit(layer_idx)
        self._swiglustep_ready = False
        self._use_ep = False

        # Router gate
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # FP32 gate weights for routing precision
        if config.need_fp32_gate:
            old_wlp = self.gate.weight.weight_loader_process
            self.gate.weight = atom_parameter(
                self.gate.weight.data.to(torch.float32)
            )
            self.gate.weight.weight_loader_process = old_wlp

        # Router bias (non-trainable, for load balancing)
        if config.use_moe_router_bias:
            self.router_bias = atom_parameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )
        else:
            self.router_bias = None

        self.renormalize = config.norm_expert_weight
        self.scoring_func = config.moe_router_activation

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=config.moe_top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_expert_weight,
            quant_config=quant_config,
            use_grouped_topk=False,
            scoring_func=config.moe_router_activation,
            e_score_correction_bias=self.router_bias,
            prefix=f"{prefix}.experts",
            has_bias=False,
            config=config,
        )

        if self.swiglu_limit > 0:
            self._swiglustep_layer_name = f"{prefix}.swiglustep"
            self._use_ep = self.experts.use_ep
            atom_config = get_current_atom_config()
            compilation_config = atom_config.compilation_config
            compilation_config.static_forward_context[
                self._swiglustep_layer_name
            ] = self
            logger.info(
                "Layer %d: swiglustep enabled (limit=%.1f, ep=%s) for routed experts",
                layer_idx, self.swiglu_limit, self._use_ep,
            )

    def _prepare_swiglustep_weights(self) -> None:
        """Store FP8 weight refs and dimensions for on-the-fly dequant.

        The bf16 buffers are NOT materialised here — they are created
        lazily the first time a CUDA-graph-captured forward runs (via
        ``_materialize_swiglustep_bf16``).  This keeps warmup memory
        ~16.9 GiB lower on TP=1, which is essential for MTP.
        """
        layer = self.experts
        self._w13_fp8 = layer.w13_weight.data
        self._w13_scale = layer.w13_weight_scale.data
        self._w2_fp8 = layer.w2_weight.data
        self._w2_scale = layer.w2_weight_scale.data

        self._inter_pad = self._w13_fp8.shape[1] // 2
        self._inter_dim = layer.intermediate_size_per_partition

        self._w13_bf16: torch.Tensor | None = None
        self._w2_bf16: torch.Tensor | None = None

        self._swiglustep_ready = True
        E = self._w13_fp8.shape[0]
        logger.info(
            "Swiglustep ready (on-the-fly dequant): %d experts, "
            "inter_pad=%d, inter_dim=%d",
            E, self._inter_pad, self._inter_dim,
        )

    def _materialize_swiglustep_bf16(self) -> None:
        """Pre-materialise bf16 expert buffers for CUDA graph capture.

        Called automatically on the first graph-captured forward.
        """
        if self._w13_bf16 is not None:
            return
        E = self._w13_fp8.shape[0]
        inter_pad = self._inter_pad
        hidden = self._w13_fp8.shape[2]
        w13_bf16 = torch.empty(E, 2 * inter_pad, hidden,
                               dtype=torch.bfloat16,
                               device=self._w13_fp8.device)
        w2_bf16 = torch.empty(E, hidden, inter_pad,
                              dtype=torch.bfloat16,
                              device=self._w2_fp8.device)
        for e in range(E):
            w13_bf16[e] = _dequant_expert(self._w13_fp8, self._w13_scale, e)
            w2_bf16[e] = _dequant_expert(self._w2_fp8, self._w2_scale, e)
        self._w13_bf16 = w13_bf16
        self._w2_bf16 = w2_bf16
        logger.info("Swiglustep bf16 materialised for graph capture: "
                    "w13=%s, w2=%s", list(w13_bf16.shape), list(w2_bf16.shape))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router logits in FP32
        router_logits = F.linear(
            hidden_states.float(), self.gate.weight.float()
        )

        if self.swiglu_limit > 0:
            routed = torch.ops.aiter.swiglustep_moe_forward(
                hidden_states, router_logits, self._swiglustep_layer_name
            )
        else:
            routed = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        if self.routed_scaling_factor != 1.0:
            routed = routed * self.routed_scaling_factor
        return routed


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Step3p5Attention(nn.Module):
    """GQA with head-wise sigmoid gating, QK norm, per-layer RoPE theta,
    per-layer partial rotary factor, and optional sliding window."""

    def __init__(
        self,
        atom_config: Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Step3p5Config = atom_config.hf_config

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        # Per-layer attention config (different Q heads for full vs sliding)
        attn_cfg = config.get_layer_attention_config(layer_idx)
        self.total_num_heads = attn_cfg["num_attention_heads"]
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = attn_cfg["num_kv_heads"]
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = attn_cfg["head_dim"]
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        sliding_window = attn_cfg["sliding_window"]
        layer_type = config.layer_types[layer_idx]

        # Head-wise gated attention: g_proj produces one scalar per Q head.
        # g_proj is a *separate* weight in the checkpoint (not interleaved in
        # q_proj), with shape [num_attention_heads, hidden_size].  We use
        # ColumnParallelLinear so the output is sharded across TP ranks
        # matching the Q head partition.
        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.g_proj = None

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-layer RoPE
        rope_theta = config.get_layer_rope_theta(layer_idx)
        partial_rotary_factor = config.get_layer_partial_rotary_factor(layer_idx)
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rope_scaling = config.get_layer_rope_scaling(layer_idx)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # QK norm
        if config.use_qk_norm:
            self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Pass rotary_emb, q_norm, k_norm into Attention so the backend
        # uses a fused norm+RoPE+cache-write kernel.  This avoids falling
        # into the unfused else-branch of rope_cache() which has a
        # scale-tensor layout mismatch (asm_layout=False with 3-D scales)
        # for sliding-window layers.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=layer_idx,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}",
            rotary_emb=self.rotary_emb,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv,
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )

        # QK norm + RoPE + KV cache write are handled inside the
        # attention backend via the fused kernel (rotary_emb / q_norm /
        # k_norm were passed to the Attention constructor).
        attn_output = self.attn(
            query=q, key=k, value=v, positions=positions, qkv=qkv
        )

        # Head-wise sigmoid gate: one scalar per head, broadcast over head_dim
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states))
            attn_output = attn_output.view(
                *attn_output.shape[:-1], self.num_heads, self.head_dim
            )
            attn_output = (
                attn_output * gate.unsqueeze(-1)
            ).view(*attn_output.shape[:-2], -1)

        output = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class Step3p5DecoderLayer(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Step3p5Config = atom_config.hf_config
        quant_config = atom_config.quant_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = Step3p5Attention(
            atom_config=atom_config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_moe = config.is_moe_layer(layer_idx)

        # Feed-forward: dense MLP or MoE + shared expert
        if self.is_moe:
            self.moe = Step3p5MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
                layer_idx=layer_idx,
            )
            swiglu_limit_shared = config.get_swiglu_limit_shared(layer_idx)
            self.share_expert = Step3p5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.share_expert_dim,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.share_expert",
                swiglu_limit=swiglu_limit_shared,
            )
            self.mlp = None
        else:
            self.moe = None
            self.share_expert = None
            self.mlp = Step3p5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

        # Layer norms (zero_centered=True → GemmaRMSNorm which uses 1+w)
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Post-attention norm + residual
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        # Feed-forward
        if self.is_moe:
            routed = self.moe(hidden_states)
            shared = self.share_expert(hidden_states)
            hidden_states = routed + shared
            if self.tp_size > 1:
                hidden_states = tensor_model_parallel_all_reduce(
                    hidden_states
                )
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@support_torch_compile
class Step3p5Model(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config: Step3p5Config = atom_config.hf_config
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Step3p5DecoderLayer(
                atom_config=atom_config,
                layer_idx=layer_num,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.moe_num_experts,
        )


# ---------------------------------------------------------------------------
# ForCausalLM
# ---------------------------------------------------------------------------


class Step3p5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # Prefixed keys prevent substring collisions with MoE stacked
        # weights (moe.gate_proj, moe.up_proj) and shared-expert weights.
        "mlp.gate_proj": ("mlp.gate_up_proj", 0),
        "mlp.up_proj": ("mlp.gate_up_proj", 1),
        "share_expert.gate_proj": ("share_expert.gate_up_proj", 0),
        "share_expert.up_proj": ("share_expert.gate_up_proj", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        self.quant_config = atom_config.quant_config

        self.model = Step3p5Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    # ------------------------------------------------------------------
    # Stacked (fused) MoE weight loading
    # ------------------------------------------------------------------
    # The checkpoint stores expert weights as single 3D tensors per layer
    # (e.g. moe.gate_proj.weight [288, 1280, 4096]) rather than per-expert
    # files.  These callbacks tell the loader how to unpack them.

    @staticmethod
    def detect_fused_expert_format(weight_name: str) -> bool:
        """Return True if *weight_name* is a stacked expert tensor."""
        return (
            ".moe.gate_proj" in weight_name
            or ".moe.up_proj" in weight_name
            or ".moe.down_proj" in weight_name
        )

    @staticmethod
    def get_fused_expert_mapping() -> list[tuple[str, str, str]]:
        return [
            ("moe.experts.w13_weight", "moe.gate_proj.weight", "w1"),
            ("moe.experts.w13_weight", "moe.up_proj.weight", "w3"),
            ("moe.experts.w2_weight", "moe.down_proj.weight", "w2"),
        ]

    @staticmethod
    def load_fused_expert_weights(
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        """Unpack a stacked [num_experts, ...] tensor into per-expert slots."""
        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = param.weight_loader
        loaded_any = False
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_any = True
            except TypeError:
                weight_loader(
                    param, loaded_weight[expert_id], name, shard_id, expert_id
                )
                loaded_any = True
        return loaded_any
