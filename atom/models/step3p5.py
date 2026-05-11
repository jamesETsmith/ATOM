# SPDX-License-Identifier: Apache-2.0
"""Inference-only Step-3.5-Flash model (stepfun-ai/Step-3.5-Flash).

196B sparse MoE (288 experts, top-8, ~11B active).  Hybrid full/sliding-window
GQA with per-head sigmoid gating, per-layer RoPE theta, and QK norms.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F

from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig
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
        x = self.act_fn(gate_up)
        if self.swiglu_limit > 0:
            x = x.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


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
    ):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.routed_scaling_factor = config.moe_router_scaling_factor

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router logits in FP32
        router_logits = F.linear(
            hidden_states.float(), self.gate.weight.float()
        )

        # Routed experts -- FusedMoE's sigmoid path does not apply
        # routed_scaling_factor internally, so we scale the output here.
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
