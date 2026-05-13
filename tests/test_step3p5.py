# SPDX-License-Identifier: MIT
# Tests for Step-3.5-Flash model support.
#
# Covers: Step3p5Config (layer-level accessors, MoE layer detection, RoPE
# config, attention head dispatch), model arch registration, packed_modules
# mapping correctness, fused-expert callback detection, and weight name
# collision avoidance.

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)

# ---------------------------------------------------------------------------
# Minimal mock so we can import Step3p5Config without heavy deps.
# Step3p5Config only requires transformers.configuration_utils.PretrainedConfig
# and transformers.AutoConfig.
# ---------------------------------------------------------------------------


class _FakePretrainedConfig:
    """Stand-in for transformers.PretrainedConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    @classmethod
    def get_config_dict(cls, *a, **kw):
        return {}, {}


class _FakeAutoConfig:
    _registry: dict = {}

    @classmethod
    def register(cls, model_type, config_class):
        cls._registry[model_type] = config_class

    @classmethod
    def for_model(cls, model_type):
        return cls._registry.get(model_type, _FakePretrainedConfig)


_mock_transformers = types.ModuleType("transformers")
_mock_transformers.AutoConfig = _FakeAutoConfig

_mock_cfg_utils = types.ModuleType("transformers.configuration_utils")
_mock_cfg_utils.PretrainedConfig = _FakePretrainedConfig

_saved_modules = {}
for _name, _mod in [
    ("transformers", _mock_transformers),
    ("transformers.configuration_utils", _mock_cfg_utils),
]:
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mod

# Now safe to import Step3p5Config (it only depends on the above).
sys.path.insert(0, ATOM_ROOT)
from atom.model_config.step3p5 import Step3p5Config  # noqa: E402

# Restore original modules
for _name, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

# ---------------------------------------------------------------------------
# Realistic config matching HF checkpoint defaults
# ---------------------------------------------------------------------------

REALISTIC_KWARGS = dict(
    vocab_size=128896,
    hidden_size=4096,
    intermediate_size=11264,
    num_hidden_layers=45,
    num_attention_heads=64,
    num_attention_groups=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=262144,
    rms_norm_eps=1e-6,
    use_qk_norm=True,
    moe_num_experts=288,
    moe_top_k=8,
    moe_intermediate_size=1280,
    share_expert_dim=1280,
    moe_layers_enum="3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44",
    moe_layer_offset=0,
    moe_every_n_layer=1,
    norm_expert_weight=True,
    moe_router_activation="sigmoid",
    moe_router_scaling_factor=3.0,
    use_moe_router_bias=True,
    need_fp32_gate=True,
    att_impl_type="GQA",
    use_head_wise_attn_gate=True,
    sliding_window=512,
    layer_types=[
        "full_attention" if i % 4 == 0 else "sliding_attention"
        for i in range(45)
    ],
    attention_other_setting={
        "attention_type": "sliding_attention",
        "num_attention_heads": 96,
        "num_attention_groups": 8,
        "head_dim": 128,
    },
    rope_theta=[5000000.0, 10000.0, 10000.0, 10000.0],
    partial_rotary_factors=[0.5, 1.0, 1.0, 1.0],
    yarn_only_types=["full_attention"],
    rope_scaling={"type": "llama3", "factor": 16.0},
    swiglu_limits=[0.0] * 43 + [7.0, 7.0],
    swiglu_limits_shared=[0.0] * 43 + [16.0, 16.0],
)


@pytest.fixture
def config():
    return Step3p5Config(**REALISTIC_KWARGS)


# ===========================================================================
# Step3p5Config tests
# ===========================================================================


class TestStep3p5ConfigLayers:
    """MoE layer detection and enumeration."""

    def test_dense_layers_are_not_moe(self, config):
        for i in range(3):
            assert not config.is_moe_layer(i), f"Layer {i} should be dense"

    def test_moe_layers_3_to_44(self, config):
        for i in range(3, 45):
            assert config.is_moe_layer(i), f"Layer {i} should be MoE"

    def test_num_experts_alias(self, config):
        assert config.num_experts == 288
        assert config.num_experts == config.moe_num_experts


class TestStep3p5ConfigAttention:
    """Per-layer attention head dispatch."""

    def test_full_attention_heads(self, config):
        attn = config.get_layer_attention_config(0)
        assert attn["num_attention_heads"] == 64
        assert attn["num_kv_heads"] == 8
        assert attn["sliding_window"] is None

    def test_sliding_attention_heads(self, config):
        attn = config.get_layer_attention_config(1)
        assert attn["num_attention_heads"] == 96
        assert attn["num_kv_heads"] == 8
        assert attn["sliding_window"] == 512

    def test_layer_type_repeats(self, config):
        for i in range(45):
            attn = config.get_layer_attention_config(i)
            if i % 4 == 0:
                assert attn["num_attention_heads"] == 64
            else:
                assert attn["num_attention_heads"] == 96


class TestStep3p5ConfigRoPE:
    """Per-layer RoPE theta and partial rotary factor."""

    def test_full_attention_theta(self, config):
        assert config.get_layer_rope_theta(0) == 5_000_000.0

    def test_sliding_attention_theta(self, config):
        assert config.get_layer_rope_theta(1) == 10_000.0
        assert config.get_layer_rope_theta(2) == 10_000.0
        assert config.get_layer_rope_theta(3) == 10_000.0

    def test_theta_cycles(self, config):
        assert config.get_layer_rope_theta(4) == 5_000_000.0
        assert config.get_layer_rope_theta(5) == 10_000.0

    def test_full_attention_partial_rotary(self, config):
        assert config.get_layer_partial_rotary_factor(0) == 0.5

    def test_sliding_attention_partial_rotary(self, config):
        assert config.get_layer_partial_rotary_factor(1) == 1.0

    def test_rope_scaling_only_full_attention(self, config):
        assert config.get_layer_rope_scaling(0) is not None
        assert config.get_layer_rope_scaling(1) is None
        assert config.get_layer_rope_scaling(4) is not None


class TestStep3p5ConfigSwiGLU:
    """SwiGLU clamp limits for late layers."""

    def test_no_limit_early_layers(self, config):
        for i in range(43):
            assert config.get_swiglu_limit(i) == 0.0

    def test_limit_layer_43(self, config):
        assert config.get_swiglu_limit(43) == 7.0

    def test_limit_layer_44(self, config):
        assert config.get_swiglu_limit(44) == 7.0

    def test_shared_limit_layer_43(self, config):
        assert config.get_swiglu_limit_shared(43) == 16.0

    def test_shared_limit_layer_44(self, config):
        assert config.get_swiglu_limit_shared(44) == 16.0


# ===========================================================================
# Model arch registration
# ===========================================================================


class TestModelRegistration:
    """Verify Step3p5ForCausalLM is in model_runner.py."""

    def test_in_support_model_arch_dict(self):
        path = os.path.join(ATOM_ROOT, "atom", "model_engine", "model_runner.py")
        with open(path) as f:
            content = f.read()
        assert '"Step3p5ForCausalLM"' in content
        assert "atom.models.step3p5.Step3p5ForCausalLM" in content

    def test_config_registered_with_autoconfig(self):
        assert "step3p5" in _FakeAutoConfig._registry
        assert _FakeAutoConfig._registry["step3p5"] is Step3p5Config


# ===========================================================================
# packed_modules_mapping correctness
# ===========================================================================


class TestPackedModulesMapping:
    """Verify packed_modules_mapping won't intercept MoE weight names."""

    @pytest.fixture
    def mapping(self):
        path = os.path.join(ATOM_ROOT, "atom", "models", "step3p5.py")
        with open(path) as f:
            src = f.read()
        # Extract the class dict by simple exec in a sandbox
        ns = {"__builtins__": __builtins__}
        # Find just the dict literal
        import re

        m = re.search(
            r"packed_modules_mapping\s*=\s*(\{.+?\})",
            src,
            re.DOTALL,
        )
        assert m, "packed_modules_mapping not found in step3p5.py"
        return eval(m.group(1))

    def test_dense_mlp_gate_proj_matches(self, mapping):
        name = "model.layers.0.mlp.gate_proj.weight"
        matched = [k for k in mapping if k in name]
        assert "mlp.gate_proj" in matched

    def test_dense_mlp_up_proj_matches(self, mapping):
        name = "model.layers.0.mlp.up_proj.weight"
        matched = [k for k in mapping if k in name]
        assert "mlp.up_proj" in matched

    def test_shared_expert_gate_proj_matches(self, mapping):
        name = "model.layers.3.share_expert.gate_proj.weight"
        matched = [k for k in mapping if k in name]
        assert "share_expert.gate_proj" in matched

    def test_shared_expert_up_proj_matches(self, mapping):
        name = "model.layers.3.share_expert.up_proj.weight"
        matched = [k for k in mapping if k in name]
        assert "share_expert.up_proj" in matched

    def test_moe_gate_proj_does_not_match(self, mapping):
        """MoE stacked weights must NOT be intercepted by packed mapping."""
        name = "model.layers.3.moe.gate_proj.weight"
        matched = [k for k in mapping if k in name]
        assert len(matched) == 0, f"Unexpected match: {matched}"

    def test_moe_up_proj_does_not_match(self, mapping):
        name = "model.layers.3.moe.up_proj.weight"
        matched = [k for k in mapping if k in name]
        assert len(matched) == 0, f"Unexpected match: {matched}"

    def test_moe_down_proj_does_not_match(self, mapping):
        name = "model.layers.3.moe.down_proj.weight"
        matched = [k for k in mapping if k in name]
        assert len(matched) == 0, f"Unexpected match: {matched}"

    def test_q_proj_matches(self, mapping):
        name = "model.layers.0.self_attn.q_proj.weight"
        matched = [k for k in mapping if k in name]
        assert "q_proj" in matched

    def test_g_proj_does_not_match(self, mapping):
        """g_proj is a separate weight; it must not match any packed key."""
        name = "model.layers.0.self_attn.g_proj.weight"
        matched = [k for k in mapping if k in name]
        assert len(matched) == 0, f"Unexpected match: {matched}"


# ===========================================================================
# Fused expert callbacks
# ===========================================================================


class TestFusedExpertCallbacks:
    """Verify detect / mapping / load callbacks exist and behave correctly."""

    def _read_model_src(self):
        path = os.path.join(ATOM_ROOT, "atom", "models", "step3p5.py")
        with open(path) as f:
            return f.read()

    def test_detect_fused_expert_format_defined(self):
        src = self._read_model_src()
        assert "def detect_fused_expert_format" in src

    def test_get_fused_expert_mapping_defined(self):
        src = self._read_model_src()
        assert "def get_fused_expert_mapping" in src

    def test_load_fused_expert_weights_defined(self):
        src = self._read_model_src()
        assert "def load_fused_expert_weights" in src

    def test_detect_matches_moe_gate_proj(self):
        src = self._read_model_src()
        # The function should detect names containing ".moe.gate_proj"
        assert '".moe.gate_proj"' in src

    def test_detect_matches_moe_up_proj(self):
        src = self._read_model_src()
        assert '".moe.up_proj"' in src

    def test_detect_matches_moe_down_proj(self):
        src = self._read_model_src()
        assert '".moe.down_proj"' in src

    def test_mapping_has_all_three_shards(self):
        src = self._read_model_src()
        assert '"w1"' in src  # gate_proj -> w1
        assert '"w2"' in src  # down_proj -> w2
        assert '"w3"' in src  # up_proj -> w3

    def test_mapping_produces_correct_params_dict_keys(self):
        """The name mapping must produce keys WITHOUT a trailing .weight suffix,
        matching the FusedMoE parameter names (e.g. 'moe.experts.w13_weight')."""
        import ast

        path = os.path.join(ATOM_ROOT, "atom", "models", "step3p5.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        mapping = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_fused_expert_mapping":
                ret = node.body[-1]
                assert isinstance(ret, ast.Return)
                mapping = ast.literal_eval(ret.value)
                break
        assert mapping is not None, "get_fused_expert_mapping not found"

        checkpoint_names = [
            "model.layers.3.moe.gate_proj.weight",
            "model.layers.3.moe.up_proj.weight",
            "model.layers.3.moe.down_proj.weight",
            "model.layers.3.moe.gate_proj.weight_scale",
            "model.layers.3.moe.up_proj.weight_scale",
            "model.layers.3.moe.down_proj.weight_scale",
        ]
        expected_mapped = {
            "model.layers.3.moe.gate_proj.weight": "model.layers.3.moe.experts.w13_weight",
            "model.layers.3.moe.up_proj.weight": "model.layers.3.moe.experts.w13_weight",
            "model.layers.3.moe.down_proj.weight": "model.layers.3.moe.experts.w2_weight",
            "model.layers.3.moe.gate_proj.weight_scale": "model.layers.3.moe.experts.w13_weight_scale",
            "model.layers.3.moe.up_proj.weight_scale": "model.layers.3.moe.experts.w13_weight_scale",
            "model.layers.3.moe.down_proj.weight_scale": "model.layers.3.moe.experts.w2_weight_scale",
        }
        for ckpt_name in checkpoint_names:
            for param_name, weight_name, shard_id in mapping:
                if weight_name in ckpt_name:
                    mapped = ckpt_name.replace(weight_name, param_name)
                    assert mapped == expected_mapped[ckpt_name], (
                        f"{ckpt_name} -> {mapped}, expected {expected_mapped[ckpt_name]}"
                    )
                    break


# ===========================================================================
# Model file structure
# ===========================================================================


class TestModelFileStructure:
    """Verify the model file has correct structure and no common mistakes."""

    @pytest.fixture
    def src(self):
        path = os.path.join(ATOM_ROOT, "atom", "models", "step3p5.py")
        with open(path) as f:
            return f.read()

    def test_no_qkvg_parallel_linear(self, src):
        """g_proj is separate; QKVGParallelLinear must not be used."""
        assert "QKVGParallelLinear" not in src

    def test_uses_column_parallel_for_g_proj(self, src):
        assert "ColumnParallelLinear" in src
        assert "g_proj" in src

    def test_uses_qkv_parallel_linear(self, src):
        assert "QKVParallelLinear" in src

    def test_uses_gemma_rms_norm(self, src):
        """zero_centered=True needs GemmaRMSNorm (1+w scaling)."""
        assert "GemmaRMSNorm" in src

    def test_passes_qk_norms_to_attention(self, src):
        """q_norm and k_norm are passed to Attention for fused cache write."""
        assert "q_norm=self.q_norm" in src
        assert "k_norm=self.k_norm" in src

    def test_sigmoid_gate_in_forward(self, src):
        assert "torch.sigmoid" in src

    def test_share_expert_at_layer_level(self, src):
        """share_expert should be a child of DecoderLayer, not MoE."""
        # Look for share_expert being set as self.share_expert in DecoderLayer
        assert "self.share_expert = Step3p5MLP" in src

    def test_moe_does_not_own_share_expert(self, src):
        """Step3p5MoE should not have share_expert as an attribute."""
        import re

        # Find the Step3p5MoE class body
        moe_match = re.search(
            r"class Step3p5MoE.*?(?=\nclass |\Z)", src, re.DOTALL
        )
        assert moe_match
        moe_body = moe_match.group(0)
        assert "self.share_expert" not in moe_body

    def test_support_torch_compile_decorator(self, src):
        assert "@support_torch_compile" in src

    def test_no_unused_imports(self, src):
        """extract_layer_index was removed as unused."""
        assert "extract_layer_index" not in src


# ===========================================================================
# MTP weight remapping
# ===========================================================================


class TestMTPWeightRemapping:
    """Verify weight name remapping for the Step3p5 MTP model."""

    @pytest.fixture
    def config(self):
        return Step3p5Config(
            **{**REALISTIC_KWARGS, "num_nextn_predict_layers": 3}
        )

    @pytest.fixture
    def remap_funcs(self):
        """Extract the pure-Python remap functions from step3p5_mtp.py
        without triggering heavy imports (torch, aiter, etc.)."""
        import ast
        import os

        path = os.path.join(ATOM_ROOT, "atom", "models", "step3p5_mtp.py")
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)
        func_sources = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_get_spec_layer_idx",
                "_rewrite_spec_layer_name",
            ):
                func_sources.append(ast.get_source_segment(source, node))

        assert len(func_sources) == 2, "Expected 2 remap functions"
        ns = {"Step3p5Config": Step3p5Config}
        for src in func_sources:
            exec(src, ns)
        return ns

    def _remap(self, remap_funcs, config, name):
        idx = remap_funcs["_get_spec_layer_idx"](config, name)
        if idx is None:
            return None
        return remap_funcs["_rewrite_spec_layer_name"](idx, name)

    def test_enorm_stays_at_layer_level(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.45.enorm.weight")
        assert result == "model.layers.45.enorm.weight"

    def test_hnorm_stays_at_layer_level(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.46.hnorm.weight")
        assert result == "model.layers.46.hnorm.weight"

    def test_eh_proj_stays_at_layer_level(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.47.eh_proj.weight")
        assert result == "model.layers.47.eh_proj.weight"

    def test_self_attn_gets_mtp_block_prefix(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.45.self_attn.q_proj.weight")
        assert result == "model.layers.45.mtp_block.self_attn.q_proj.weight"

    def test_mlp_gets_mtp_block_prefix(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.46.mlp.gate_proj.weight")
        assert result == "model.layers.46.mtp_block.mlp.gate_proj.weight"

    def test_input_layernorm_gets_mtp_block_prefix(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.45.input_layernorm.weight")
        assert result == "model.layers.45.mtp_block.input_layernorm.weight"

    def test_post_attn_layernorm_gets_mtp_block_prefix(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.47.post_attention_layernorm.weight")
        assert result == "model.layers.47.mtp_block.post_attention_layernorm.weight"

    def test_shared_head_norm_remapped(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.45.transformer.shared_head.norm.weight")
        assert result == "model.layers.45.shared_head.norm.weight"

    def test_shared_head_output_remapped_to_head(self, config, remap_funcs):
        result = self._remap(remap_funcs, config, "model.layers.45.transformer.shared_head.output.weight")
        assert result == "model.layers.45.shared_head.head.weight"

    def test_top_level_embed_tokens_passthrough(self, config, remap_funcs):
        # The Step-3.5-Flash checkpoint stores embed_tokens at top level
        # (model.embed_tokens.weight); the MTP container reuses it as a
        # shared weight and the name should pass through unchanged.
        idx = remap_funcs["_get_spec_layer_idx"](config, "model.embed_tokens.weight")
        assert idx is None  # not under model.layers.{45..47}
        # Even when applied directly, the layer-prefixed form is not a
        # real checkpoint weight; verify embed_tokens at top level passes through.
        result = remap_funcs["_rewrite_spec_layer_name"](45, "model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight"

    def test_non_mtp_layer_returns_none(self, config, remap_funcs):
        idx = remap_funcs["_get_spec_layer_idx"](config, "model.layers.44.self_attn.q_proj.weight")
        assert idx is None

    def test_all_checkpoint_weights_remap(self, config, remap_funcs):
        """Every real checkpoint weight name for MTP layers should remap."""
        checkpoint_names = [
            "model.layers.45.eh_proj.weight",
            "model.layers.45.enorm.weight",
            "model.layers.45.hnorm.weight",
            "model.layers.45.input_layernorm.weight",
            "model.layers.45.mlp.down_proj.weight",
            "model.layers.45.mlp.gate_proj.weight",
            "model.layers.45.mlp.up_proj.weight",
            "model.layers.45.post_attention_layernorm.weight",
            "model.layers.45.self_attn.g_proj.weight",
            "model.layers.45.self_attn.k_norm.weight",
            "model.layers.45.self_attn.k_proj.weight",
            "model.layers.45.self_attn.o_proj.weight",
            "model.layers.45.self_attn.q_norm.weight",
            "model.layers.45.self_attn.q_proj.weight",
            "model.layers.45.self_attn.v_proj.weight",
            "model.layers.45.transformer.shared_head.norm.weight",
            "model.layers.45.transformer.shared_head.output.weight",
        ]
        for name in checkpoint_names:
            result = self._remap(remap_funcs, config, name)
            assert result is not None, f"Failed to remap: {name}"
            assert result != name or "enorm" in name or "hnorm" in name or "eh_proj" in name, (
                f"Unexpected no-op remap: {name} -> {result}"
            )


# ===========================================================================
# MTP registration
# ===========================================================================


class TestMTPRegistration:
    """Verify MTP model is registered in config.py and eagle.py."""

    def test_mtp_type_map_has_step3p5(self):
        path = os.path.join(ATOM_ROOT, "atom", "config.py")
        with open(path) as f:
            content = f.read()
        assert '"step3p5": "step3p5_mtp"' in content

    def test_mtp_config_has_step3p5_mtp(self):
        path = os.path.join(ATOM_ROOT, "atom", "config.py")
        with open(path) as f:
            content = f.read()
        assert '"step3p5_mtp"' in content
        assert '"Step3p5MTPModel"' in content

    def test_eagle_arch_dict_has_step3p5(self):
        path = os.path.join(ATOM_ROOT, "atom", "spec_decode", "eagle.py")
        with open(path) as f:
            content = f.read()
        assert '"Step3p5MTPModel"' in content
        assert "atom.models.step3p5_mtp.Step3p5MTP" in content
