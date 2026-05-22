# SPDX-License-Identifier: Apache-2.0
"""Step-3.5-Flash model configuration (stepfun-ai/Step-3.5-Flash)."""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class Step3p5Config(PretrainedConfig):
    """Configuration for Step-3.5-Flash sparse MoE model.

    Mirrors the HF-hosted ``configuration_step3p5.py`` so that ATOM can
    load the model without ``trust_remote_code=True``.
    """

    model_type = "step3p5"

    def __init__(
        self,
        vocab_size: int = 128896,
        hidden_size: int = 4096,
        intermediate_size: int = 11264,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_attention_groups: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        max_seq_len: int = 262144,
        rms_norm_eps: float = 1e-5,
        use_qk_norm: bool = True,
        tie_word_embeddings: bool = False,
        # MoE
        use_moe: bool = True,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        moe_intermediate_size: int = 1280,
        share_expert_dim: int = 1280,
        moe_layers_enum: str = "",
        moe_layer_offset: int = 0,
        moe_every_n_layer: int = 1,
        norm_expert_weight: bool = True,
        moe_router_activation: str = "sigmoid",
        moe_router_scaling_factor: float = 3.0,
        use_moe_router_bias: bool = True,
        need_fp32_gate: bool = True,
        # Attention
        att_impl_type: str = "GQA",
        use_head_wise_attn_gate: bool = True,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        attention_other_setting: dict | None = None,
        # RoPE
        rope_scaling: dict | None = None,
        rope_theta: list[float] | float | None = None,
        partial_rotary_factors: list[float] | None = None,
        yarn_only_types: list[str] | None = None,
        use_rope_layers: list[int] | None = None,
        # MTP
        num_nextn_predict_layers: int = 0,
        # Other
        zero_centered: bool = True,
        sink: bool = False,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm

        # MoE
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_layer_offset = moe_layer_offset
        self.moe_every_n_layer = moe_every_n_layer
        self.norm_expert_weight = norm_expert_weight
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate

        # Parse moe_layers_enum: comma-separated layer indices
        self.moe_layers_enum = moe_layers_enum
        self._moe_layers: set | None = None

        # Attention
        self.att_impl_type = att_impl_type
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.sliding_window = sliding_window
        self.layer_types = layer_types or [
            "full_attention" if i % 4 == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        self.attention_other_setting = attention_other_setting or {
            "attention_type": "sliding_attention",
            "num_attention_heads": 96,
            "num_attention_groups": 8,
            "head_dim": 128,
        }

        # RoPE
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta if rope_theta is not None else 10000.0
        self.partial_rotary_factors = partial_rotary_factors
        self.yarn_only_types = yarn_only_types or ["full_attention"]
        self.use_rope_layers = use_rope_layers or []

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Other
        self.zero_centered = zero_centered
        self.sink = sink
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared

    @property
    def num_key_value_heads(self) -> int:
        """KV head count expected by the attention backend and KV cache allocator.

        Step 3.5 uses ``num_attention_groups`` for KV heads.  Both
        full-attention and sliding-attention layers share the same KV
        head count (8), so a single flat value is correct.
        """
        return self.num_attention_groups

    @property
    def num_experts(self) -> int:
        """Alias used by the weight loader."""
        return self.moe_num_experts

    @property
    def routed_scaling_factor(self) -> float:
        """Alias expected by ``FusedMoE`` for routing weight scaling."""
        return self.moe_router_scaling_factor

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return True if the given layer uses MoE instead of dense MLP."""
        if self._moe_layers is None:
            if isinstance(self.moe_layers_enum, str) and self.moe_layers_enum.strip():
                self._moe_layers = set(
                    int(x) for x in self.moe_layers_enum.split(",")
                )
            else:
                self._moe_layers = set()
        return layer_idx in self._moe_layers

    def get_layer_attention_config(self, layer_idx: int) -> dict:
        """Return attention head config for a given layer index."""
        if layer_idx < len(self.layer_types):
            layer_type = self.layer_types[layer_idx]
        else:
            layer_type = "sliding_attention"
        if layer_type == "sliding_attention":
            return {
                "num_attention_heads": self.attention_other_setting[
                    "num_attention_heads"
                ],
                "num_kv_heads": self.attention_other_setting["num_attention_groups"],
                "head_dim": self.attention_other_setting.get(
                    "head_dim", self.head_dim
                ),
                "sliding_window": self.sliding_window,
            }
        else:
            return {
                "num_attention_heads": self.num_attention_heads,
                "num_kv_heads": self.num_attention_groups,
                "head_dim": self.head_dim,
                "sliding_window": None,
            }

    def get_layer_rope_theta(self, layer_idx: int) -> float:
        """Return RoPE theta for a given layer."""
        if isinstance(self.rope_theta, list):
            return self.rope_theta[layer_idx % len(self.rope_theta)]
        return self.rope_theta

    def get_layer_partial_rotary_factor(self, layer_idx: int) -> float:
        """Return partial rotary factor for a given layer."""
        if self.partial_rotary_factors is not None:
            return self.partial_rotary_factors[layer_idx % len(self.partial_rotary_factors)]
        return 1.0

    def get_layer_rope_scaling(self, layer_idx: int) -> dict | None:
        """Return RoPE scaling config for a given layer.

        Only full-attention layers use YaRN scaling per ``yarn_only_types``.
        """
        if self.rope_scaling is None:
            return None
        if layer_idx < len(self.layer_types):
            layer_type = self.layer_types[layer_idx]
        else:
            layer_type = "sliding_attention"
        if layer_type in self.yarn_only_types:
            return self.rope_scaling
        return None

    def get_swiglu_limit(self, layer_idx: int) -> float:
        """Return SwiGLU clamp limit for routed experts at a given layer."""
        if self.swiglu_limits and layer_idx < len(self.swiglu_limits):
            return self.swiglu_limits[layer_idx]
        return 0.0

    def get_swiglu_limit_shared(self, layer_idx: int) -> float:
        """Return SwiGLU clamp limit for the shared expert at a given layer."""
        if self.swiglu_limits_shared and layer_idx < len(self.swiglu_limits_shared):
            return self.swiglu_limits_shared[layer_idx]
        return 0.0


AutoConfig.register("step3p5", Step3p5Config)

__all__ = ["Step3p5Config"]
