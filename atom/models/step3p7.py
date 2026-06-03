# SPDX-License-Identifier: Apache-2.0
"""Inference-only Step-3.7-Flash (text-only path).

Step-3.7-Flash is a VLM checkpoint whose language backbone uses the same
``step3p5`` HF config and flat ``model.*`` weight layout as Step-3.5-Flash.
Subclass :class:`Step3p5ForCausalLM` so checkpoint keys map to ``model.*``
(not ``model.model.*``). Vision weights are skipped via
:pyattr:`skip_weight_prefixes`.
"""

from atom.models.step3p5 import Step3p5ForCausalLM


class Step3p7ForCausalLM(Step3p5ForCausalLM):
    """Step-3.7-Flash text-only: same module tree as Step-3.5-Flash."""

    skip_weight_prefixes = [
        "vision_model.",
        "vit_large_projector.",
    ]
