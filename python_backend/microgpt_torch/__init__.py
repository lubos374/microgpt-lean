from __future__ import annotations

from .checkpoint import load_lean_checkpoint, read_lean_checkpoint, save_lean_checkpoint, write_lean_checkpoint
from .config import GPTConfigTorch, load_config_json, save_config_json

__all__ = [
    "GPTConfigTorch",
    "load_config_json",
    "load_lean_checkpoint",
    "read_lean_checkpoint",
    "save_config_json",
    "save_lean_checkpoint",
    "write_lean_checkpoint",
]

try:
    from .generate import generate_text, generate_token_ids, sample_next_token
    from .model import MicroGPTTorch
    from .parity import (
        DEFAULT_FIXTURE_CASES,
        FIXTURE_CONFIGS,
        LeanParityBridge,
        ParityResult,
        compare_logits,
        cross_entropy_from_logits,
        run_config_parity_case,
        run_parity_case,
    )
    from .train import (
        TokenExample,
        build_priority_training_texts,
        split_byte_token_examples_from_texts,
        train_byte_model,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - local env may not have torch installed
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "MicroGPTTorch",
            "generate_text",
            "generate_token_ids",
            "sample_next_token",
            "DEFAULT_FIXTURE_CASES",
            "FIXTURE_CONFIGS",
            "LeanParityBridge",
            "ParityResult",
            "compare_logits",
            "cross_entropy_from_logits",
            "run_config_parity_case",
            "run_parity_case",
            "TokenExample",
            "build_priority_training_texts",
            "split_byte_token_examples_from_texts",
            "train_byte_model",
        ]
    )
