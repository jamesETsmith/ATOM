#!/usr/bin/env python3
"""Diagnostic: compare ATOM's Step3p5 hidden states layer-by-layer against vLLM.

This script runs inside the ATOM venv and inside the vLLM docker to dump
hidden states at various checkpoints during forward pass.
"""
import argparse
import json
import os
import sys

MODEL_CT = "/models/Step-3.5-Flash-FP8"
MODEL_HOST = "/data/jamesmit/models/Step-3.5-Flash-FP8"

# A single short prompt for diagnostics
PROMPT = "What is 2+2?"


def run_vllm_diagnostic():
    """Run inside vLLM docker: hook model to dump hidden states."""
    import torch
    from vllm import LLM, SamplingParams

    # Create LLM
    llm = LLM(
        model=MODEL_CT,
        tensor_parallel_size=1,
        trust_remote_code=False,
        gpu_memory_utilization=0.92,
        kv_cache_dtype="fp8",
        max_model_len=16384,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    # Get the model instance
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Hook to capture hidden states
    captured = {}

    # Hook embedding output
    orig_forward = model.model.forward.__func__

    def hooked_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if hasattr(self, '_is_first_rank') or True:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)

            captured["input_ids"] = input_ids.cpu().tolist()
            captured["positions"] = positions.cpu().tolist()
            captured["embed"] = {
                "shape": list(hidden_states.shape),
                "mean": float(hidden_states.float().mean()),
                "std": float(hidden_states.float().std()),
                "norm": float(hidden_states.float().norm()),
                "first_5": hidden_states[0, :5].float().cpu().tolist(),
                "last_5": hidden_states[0, -5:].float().cpu().tolist(),
            }

            for i in range(self.start_layer, min(self.end_layer, self.start_layer + 3)):
                layer = self.layers[i]
                hidden_states = layer(positions, hidden_states)
                captured[f"layer_{i}"] = {
                    "shape": list(hidden_states.shape),
                    "mean": float(hidden_states.float().mean()),
                    "std": float(hidden_states.float().std()),
                    "norm": float(hidden_states.float().norm()),
                    "first_5": hidden_states[0, :5].float().cpu().tolist(),
                    "last_5": hidden_states[0, -5:].float().cpu().tolist(),
                }

            # Continue with remaining layers
            for i in range(self.start_layer + 3, self.end_layer):
                layer = self.layers[i]
                hidden_states = layer(positions, hidden_states)

            captured["pre_norm"] = {
                "mean": float(hidden_states.float().mean()),
                "std": float(hidden_states.float().std()),
                "norm": float(hidden_states.float().norm()),
                "first_5": hidden_states[0, :5].float().cpu().tolist(),
                "last_5": hidden_states[0, -5:].float().cpu().tolist(),
            }

            return hidden_states
        return orig_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    import types
    model.model.forward = types.MethodType(hooked_forward, model.model)

    # Run inference
    sp = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    outputs = llm.generate([PROMPT], sp, use_tqdm=False)

    # Get argmax token
    output = outputs[0]
    if output.outputs:
        captured["argmax_token"] = output.outputs[0].token_ids[0] if output.outputs[0].token_ids else -1
    captured["prompt_tokens"] = list(output.prompt_token_ids)

    return captured


def run_atom_diagnostic():
    """Run inside ATOM venv: hook model to dump hidden states."""
    import torch
    import os
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")

    # We need to set up distributed env first
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from atom.config import Config
    from atom.model_config.step3p5 import Step3p5Config
    from atom.models.step3p5 import Step3p5ForCausalLM
    from transformers import AutoTokenizer

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HOST, trust_remote_code=False)
    token_ids = tokenizer.encode(PROMPT)

    captured = {"prompt_tokens": token_ids, "prompt": PROMPT}

    # Try to load model and run forward manually
    # This requires setting up distributed environment
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=1, rank=0)

        from aiter.dist.parallel_state import (
            ensure_model_parallel_initialized,
            model_parallel_is_initialized,
        )
        if not model_parallel_is_initialized():
            ensure_model_parallel_initialized(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

        # Create config
        config = Config(
            model=MODEL_HOST,
            trust_remote_code=False,
            kv_cache_dtype="fp8",
            enforce_eager=True,
        )

        # Create model
        model = Step3p5ForCausalLM(atom_config=config)
        model = model.to("cuda:0")

        # Load weights
        from atom.model_loader.loader import load_model
        load_model(model, config)

        model.eval()

        # Prepare input
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
        positions = torch.arange(len(token_ids), dtype=torch.long, device="cuda:0").unsqueeze(0)

        captured["input_ids"] = input_ids.cpu().tolist()[0]
        captured["positions"] = positions.cpu().tolist()[0]

        # Get embedding
        with torch.no_grad():
            embed = model.model.get_input_embeddings(input_ids.squeeze(0))
            captured["embed"] = {
                "shape": list(embed.shape),
                "mean": float(embed.float().mean()),
                "std": float(embed.float().std()),
                "norm": float(embed.float().norm()),
                "first_5": embed[0, :5].float().cpu().tolist() if embed.dim() > 1 else embed[:5].float().cpu().tolist(),
            }

        captured["status"] = "embedding_computed"

    except Exception as e:
        captured["error"] = str(e)
        import traceback
        captured["traceback"] = traceback.format_exc()

    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["atom", "vllm"])
    args = parser.parse_args()

    if args.engine == "vllm":
        result = run_vllm_diagnostic()
    else:
        result = run_atom_diagnostic()

    print("__DIAG_JSON_BEGIN__")
    print(json.dumps(result, indent=2))
    print("__DIAG_JSON_END__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
