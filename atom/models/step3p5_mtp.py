# SPDX-License-Identifier: Apache-2.0
"""Inference-only Step-3.5-Flash MTP (Multi-Token Prediction) model.

Each MTP layer (checkpoint layers 45-47) contains:
  - enorm / hnorm: GemmaRMSNorm for embedding and hidden state inputs
  - eh_proj: Linear(hidden*2 → hidden, no bias)
  - A full Step3p5DecoderLayer (dense MLP, sliding attention with 96 Q heads)
  - shared_head: per-layer GemmaRMSNorm + ParallelLMHead

The MTP layers are NOT MoE — they use dense MLP with intermediate_size=11264.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from atom.config import Config, QuantizationConfig
from atom.model_config.step3p5 import Step3p5Config
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.models.utils import IntermediateTensors

from atom.utils.decorators import support_torch_compile

from .step3p5 import Step3p5DecoderLayer
from .utils import maybe_prefix


class Step3p5SharedHead(nn.Module):
    """Per-layer output head: GemmaRMSNorm + ParallelLMHead.

    Checkpoint weight paths (relative to MTP layer prefix):
      - ``shared_head.norm.weight``
      - ``shared_head.head.weight`` (mapped from ``transformer.shared_head.output.weight``)
    """

    def __init__(
        self,
        config: Step3p5Config,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Step3p5MultiTokenPredictorLayer(nn.Module):
    """Single MTP layer: enorm/hnorm → eh_proj → decoder_layer → shared_head."""

    def __init__(self, atom_config: Config, prefix: str, layer_idx: int) -> None:
        super().__init__()
        config: Step3p5Config = atom_config.hf_config
        self.config = config

        self.enorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        self.shared_head = Step3p5SharedHead(
            config=config,
            prefix=maybe_prefix(prefix, "shared_head"),
            quant_config=atom_config.quant_config,
        )

        # Pass prefix that includes ".mtp_block" so child layer paths
        # match the checkpoint's quantization_config.modules_to_not_convert
        # entries (which look like "model.layers.45.mtp_block.self_attn.g_proj").
        # MTP weights in Step-3.5-Flash-FP8 are stored as BF16, so they MUST
        # land in the exclude_layers path or FP8 GEMM kernels reject them.
        self.mtp_block = Step3p5DecoderLayer(
            atom_config=atom_config,
            layer_idx=layer_idx,
            prefix=maybe_prefix(prefix, "mtp_block"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        masked_inputs_embeds = inputs_embeds
        inputs_embeds = self.enorm(masked_inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states
        return hidden_states


class Step3p5MultiTokenPredictor(nn.Module):
    """Container for the active MTP layer(s).

    The Step-3.5-Flash checkpoint stores 3 distinct MTP layers
    (model.layers.45/46/47), but ATOM's SpeculativeConfig overrides
    ``num_nextn_predict_layers`` to 1 (a single layer reused per spec step).
    We therefore only instantiate the layers that will actually run, which
    keeps both the parameter count and the KV-cache binding loop in sync
    with what allocate_kv_cache() reserves.
    """

    def __init__(self, *, atom_config: Config, prefix: str = ""):
        super().__init__()
        config: Step3p5Config = atom_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        # Prefer the draft config's (overridden) value over the target config:
        # SpeculativeConfig.hf_config_override forces it to 1 so we only build
        # the layers we will actually use.
        spec_cfg = atom_config.speculative_config
        if spec_cfg is not None and spec_cfg.draft_model_hf_config is not None:
            self.num_mtp_layers = getattr(
                spec_cfg.draft_model_hf_config,
                "num_nextn_predict_layers",
                1,
            )
        else:
            self.num_mtp_layers = config.num_nextn_predict_layers

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Step3p5MultiTokenPredictorLayer(
                    atom_config, f"{prefix}.layers.{idx}", layer_idx=idx
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = mtp_layer.shared_head.head(mtp_layer.shared_head(hidden_states))
        return logits


@support_torch_compile
class Step3p5MTP(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "mlp.gate_proj": ("mlp.gate_up_proj", 0),
        "mlp.up_proj": ("mlp.gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config: Step3p5Config = atom_config.hf_config

        self.model = Step3p5MultiTokenPredictor(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "model")
        )

    def remap_mtp_weight_name(self, name: str) -> str | None:
        spec_layer = _get_spec_layer_idx(self.config, name)
        if spec_layer is None:
            return None
        # Only the first ``num_mtp_layers`` MTP layers are instantiated
        # (typically 1 because SpeculativeConfig overrides the count); skip
        # weights for any layers we did not build, otherwise the loader
        # raises KeyError trying to find a non-existent module.
        active_count = self.model.num_mtp_layers
        mtp_start = self.config.num_hidden_layers
        if spec_layer >= mtp_start + active_count:
            return None
        return _rewrite_spec_layer_name(spec_layer, name)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return []


# ---------------------------------------------------------------------------
# Weight name remapping utilities
# ---------------------------------------------------------------------------


def _get_spec_layer_idx(config: Step3p5Config, weight_name: str) -> int | None:
    """Return the layer index if ``weight_name`` belongs to an MTP layer."""
    if config.num_nextn_predict_layers <= 0:
        return None
    base = config.num_hidden_layers
    for i in range(config.num_nextn_predict_layers):
        if weight_name.startswith(f"model.layers.{base + i}."):
            return base + i
    return None


def _rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
    """Rewrite checkpoint weight names for the MTP module structure.

    Checkpoint layout (Step-3.5-Flash, layers 45-47):
      model.layers.{idx}.enorm.weight
      model.layers.{idx}.hnorm.weight
      model.layers.{idx}.eh_proj.weight
      model.layers.{idx}.self_attn.{g,q,k,v,o}_proj.weight
      model.layers.{idx}.self_attn.{q,k}_norm.weight
      model.layers.{idx}.mlp.{gate,up,down}_proj.weight
      model.layers.{idx}.input_layernorm.weight
      model.layers.{idx}.post_attention_layernorm.weight
      model.layers.{idx}.transformer.shared_head.norm.weight
      model.layers.{idx}.transformer.shared_head.output.weight
      model.embed_tokens.weight (shared with main model)

    Module layout in this file places the decoder block under ``.mtp_block``
    so its child layer paths match the checkpoint's
    ``quantization_config.modules_to_not_convert`` entries (which DO contain
    ``mtp_block``, e.g. ``model.layers.45.mtp_block.self_attn.g_proj``).

    Rules:
    * ``model.embed_tokens`` → kept at top level (shared with main model)
    * ``transformer.shared_head.norm`` → ``shared_head.norm``
    * ``transformer.shared_head.output`` → ``shared_head.head``
    * ``enorm``/``hnorm``/``eh_proj`` → kept at layer level (no rewrite)
    * Everything else (self_attn, mlp, layernorms) → ``.mtp_block.`` injected
    """
    spec_layer_prefix = f"model.layers.{spec_layer}."

    # Handle transformer.shared_head → shared_head and output → head
    if ".transformer.shared_head." in name:
        name = name.replace(
            f"{spec_layer_prefix}transformer.shared_head.",
            f"{spec_layer_prefix}shared_head.",
        )
        name = name.replace(".shared_head.output.", ".shared_head.head.")
        return name

    # Shared embedding lives at the top level of the MTP container.
    if name.startswith("model.embed_tokens."):
        return name

    # Layer-level weights that are NOT inside the decoder block.
    spec_layer_weight_names = ("enorm", "hnorm", "eh_proj", "shared_head")
    for wn in spec_layer_weight_names:
        if f"{spec_layer_prefix}{wn}" in name:
            return name

    # Everything else (self_attn, mlp, input_layernorm, post_attention_layernorm)
    # belongs inside the mtp_block submodule.
    return name.replace(spec_layer_prefix, f"{spec_layer_prefix}mtp_block.")
