# SPDX-License-Identifier: Apache-2.0
"""Inference-only Step-3.7-Flash (text-only path).

Step-3.7-Flash is a VLM checkpoint whose language backbone uses the same
``step3p5`` HF config and flat ``model.*`` weight layout as Step-3.5-Flash.
This wrapper loads only the language tensors and skips vision encoder and
projector weights via :pyattr:`skip_weight_prefixes`.
"""

from typing import Optional, Union

import torch
from torch import nn

from atom.config import Config
from atom.models.step3p5 import Step3p5ForCausalLM
from atom.models.utils import IntermediateTensors


class Step3p7ForCausalLM(nn.Module):
    """Step-3.7-Flash text-only wrapper around :class:`Step3p5ForCausalLM`.

    Checkpoint language weights use ``model.*`` / ``lm_head.*`` (not
    ``language_model.*``), so the inner model is named ``model`` to match
    safetensors keys directly.
    """

    skip_weight_prefixes = [
        "vision_model.",
        "vit_large_projector.",
    ]

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = atom_config.hf_config
        self.model = Step3p5ForCausalLM(atom_config=atom_config, prefix=prefix)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @property
    def packed_modules_mapping(self):
        return self.model.packed_modules_mapping

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    @staticmethod
    def detect_fused_expert_format(weight_name: str) -> bool:
        return Step3p5ForCausalLM.detect_fused_expert_format(weight_name)

    @staticmethod
    def get_fused_expert_mapping() -> list[tuple[str, str, int, str]]:
        return Step3p5ForCausalLM.get_fused_expert_mapping()

    @staticmethod
    def load_fused_expert_weights(
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        return Step3p5ForCausalLM.load_fused_expert_weights(
            original_name,
            name,
            params_dict,
            loaded_weight,
            shard_id,
            num_experts,
        )
