# SPDX-License-Identifier: MIT
# Tests for Step-3.7-Flash model support (text-only path).

import contextlib
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)


class QuantType(enum.IntEnum):
    No = 0


class FakeHFConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def get_config_dict(model, **kwargs):
        return {}, {}


class FakeAutoConfig:
    _registry: dict = {}

    @classmethod
    def for_model(cls, model_type):
        return cls

    @classmethod
    def from_dict(cls, d):
        return FakeHFConfig(**d)

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return FakeHFConfig(model_type=model)


@contextlib.contextmanager
def _temporary_mocks():
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "torch.bfloat16"

    mock_aiter = types.ModuleType("aiter")
    mock_aiter.QuantType = QuantType
    mock_aiter.__path__ = []

    mock_aiter_dtypes = types.ModuleType("aiter.utility.dtypes")
    mock_aiter_dtypes.d_dtypes = {}

    mock_transformers = types.ModuleType("transformers")
    mock_transformers.PretrainedConfig = FakeHFConfig
    mock_transformers.AutoConfig = FakeAutoConfig
    mock_transformers.GenerationConfig = MagicMock()

    mock_atom_utils = types.ModuleType("atom.utils")
    mock_atom_utils.envs = MagicMock()
    mock_atom_utils.get_open_port = MagicMock(return_value=8000)

    mock_dist_utils = types.ModuleType("atom.utils.distributed.utils")
    mock_dist_utils.stateless_init_torch_distributed_process_group = MagicMock()

    mock_plugin = types.ModuleType("atom.plugin")
    mock_plugin.is_plugin_mode = MagicMock(return_value=False)
    mock_plugin.is_vllm = MagicMock(return_value=False)
    mock_plugin_config = types.ModuleType("atom.plugin.config")
    mock_plugin_config.PluginConfig = MagicMock()

    mock_quant_spec = types.ModuleType("atom.quant_spec")
    mock_quant_spec.LayerQuantConfig = MagicMock()
    mock_quant_spec.get_quant_parser = MagicMock(return_value=MagicMock())

    patches = {
        "torch": mock_torch,
        "torch.distributed": MagicMock(),
        "aiter": mock_aiter,
        "aiter.utility": types.ModuleType("aiter.utility"),
        "aiter.utility.dtypes": mock_aiter_dtypes,
        "transformers": mock_transformers,
        "atom.utils": mock_atom_utils,
        "atom.utils.distributed": types.ModuleType("atom.utils.distributed"),
        "atom.utils.distributed.utils": mock_dist_utils,
        "atom.plugin": mock_plugin,
        "atom.plugin.config": mock_plugin_config,
        "atom.quant_spec": mock_quant_spec,
    }

    saved = {}
    for name in list(sys.modules):
        if name == "aiter" or name.startswith("aiter."):
            saved[name] = sys.modules.pop(name, None)
    for name, mock in patches.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock
    try:
        yield
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _load_config():
    path = os.path.join(ATOM_ROOT, "atom", "config.py")
    spec = importlib.util.spec_from_file_location("_atom_config_step37_test", path)
    mod = importlib.util.module_from_spec(spec)
    with _temporary_mocks():
        spec.loader.exec_module(mod)
    return mod


_m = _load_config()
_MULTIMODAL_MODEL_TYPES = _m._MULTIMODAL_MODEL_TYPES
_ATOM_CONFIG_CLASSES = _m._ATOM_CONFIG_CLASSES
get_hf_config = _m.get_hf_config


STEP37_TEXT_CONFIG = {
    "model_type": "step3p5",
    "hidden_size": 4096,
    "num_hidden_layers": 45,
    "moe_num_experts": 288,
    "moe_top_k": 8,
    "moe_layers_enum": "3,4,5",
}


class TestConfigRegistry:
    def test_step3p7_in_multimodal_registry(self):
        assert "step3p7" in _MULTIMODAL_MODEL_TYPES
        assert _MULTIMODAL_MODEL_TYPES["step3p7"] == "text_config"

    def test_step3p5_in_atom_config_classes(self):
        assert "step3p5" in _ATOM_CONFIG_CLASSES


class TestGetHfConfigMultimodal:
    def _make_config_dict(self, **overrides):
        base = {
            "model_type": "step3p7",
            "architectures": ["Step3p7ForConditionalGeneration"],
            "bos_token_id": 0,
            "eos_token_id": [1, 2, 128007],
            "quantization_config": {"quant_method": "fp8"},
            "text_config": dict(STEP37_TEXT_CONFIG),
        }
        base.update(overrides)
        return base

    @contextlib.contextmanager
    def _patch_get_config_dict(self, config_dict):
        original = FakeHFConfig.get_config_dict

        @staticmethod
        def patched(model, **kwargs):
            return config_dict, {}

        FakeHFConfig.get_config_dict = patched
        try:
            yield
        finally:
            FakeHFConfig.get_config_dict = original

    def test_extracts_text_config(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("stepfun-ai/Step-3.7-Flash-FP8")

        assert hf_config.hidden_size == 4096
        assert hf_config.num_hidden_layers == 45
        assert hf_config.moe_num_experts == 288

    def test_preserves_step3p7_architecture(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("stepfun-ai/Step-3.7-Flash-FP8")

        assert hf_config.architectures == ["Step3p7ForConditionalGeneration"]

    def test_propagates_eos_token_id_list(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("stepfun-ai/Step-3.7-Flash-FP8")

        assert hf_config.eos_token_id == [1, 2, 128007]

    def test_propagates_quantization_config(self):
        config_dict = self._make_config_dict()
        with self._patch_get_config_dict(config_dict):
            hf_config = get_hf_config("stepfun-ai/Step-3.7-Flash-FP8")

        assert hf_config.quantization_config["quant_method"] == "fp8"


class TestModelRegistration:
    def test_step3p7_in_support_model_arch_dict(self):
        path = os.path.join(ATOM_ROOT, "atom", "model_engine", "model_runner.py")
        with open(path) as f:
            content = f.read()
        assert '"Step3p7ForConditionalGeneration"' in content
        assert "atom.models.step3p7.Step3p7ForCausalLM" in content


class TestStep3p7Wrapper:
    def test_skip_weight_prefixes_defined(self):
        path = os.path.join(ATOM_ROOT, "atom", "models", "step3p7.py")
        with open(path) as f:
            content = f.read()
        assert "vision_model." in content
        assert "vit_large_projector." in content

    def test_loader_reads_skip_weight_prefixes(self):
        path = os.path.join(ATOM_ROOT, "atom", "model_loader", "loader.py")
        with open(path) as f:
            content = f.read()
        assert 'getattr(model, "skip_weight_prefixes", [])' in content


class TestLoaderSkipLogic:
    def test_vision_prefix_would_be_skipped(self):
        prefixes = ["vision_model.", "vit_large_projector."]
        names = [
            "vision_model.conv1.weight",
            "vit_large_projector.weight",
            "model.embed_tokens.weight",
        ]
        for name in names:
            skipped = any(name.startswith(p) for p in prefixes)
            if name.startswith("model."):
                assert not skipped
            else:
                assert skipped
