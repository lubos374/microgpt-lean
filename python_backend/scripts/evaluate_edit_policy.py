from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.checkpoint import load_lean_checkpoint
from microgpt_torch.config import GPTConfigTorch, load_config_json
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.train import byte_config, encode_utf8_bytes
from self_edit.schemas import EditPromptExample


@dataclass(frozen=True, slots=True)
class EditPolicyMetric:
    examples: int
    response_tokens: int
    loss: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint as a declaration-edit proposer over prompt/response pairs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL edit dataset from build_edit_dataset.py")
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint.parent/config.json if present.")
    parser.add_argument("--split", choices=["train", "valid", "all"], default="valid")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def resolve_config(checkpoint: Path, config_json: Path | None) -> GPTConfigTorch:
    if config_json is not None:
        return load_config_json(config_json)
    inferred = checkpoint.parent / "config.json"
    if inferred.exists():
        return load_config_json(inferred)
    return byte_config()


def iter_examples(path: Path, split: str, limit: int | None) -> list[EditPromptExample]:
    examples: list[EditPromptExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if split != "all" and row["split"] != split:
                continue
            examples.append(
                EditPromptExample(
                    split=row["split"],
                    source_id=row["source_id"],
                    prompt=row["prompt"],
                    response=row["response"],
                    target=row["target"],
                )
            )
            if limit is not None and limit > 0 and len(examples) >= limit:
                break
    return examples


@torch.no_grad()
def evaluate_edit_examples(
    model: MicroGPTTorch,
    examples: Iterable[EditPromptExample],
    *,
    context: int,
    device: torch.device | str | None = None,
) -> EditPolicyMetric:
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_examples = 0
    device = model.embedding.device if device is None else device
    for example in examples:
        prompt_ids = encode_utf8_bytes(example.prompt)
        response_ids = encode_utf8_bytes(example.response)
        if not prompt_ids or not response_ids:
            continue
        full = prompt_ids + response_ids
        prompt_len = len(prompt_ids)
        example_tokens = 0
        for target_pos in range(prompt_len, len(full)):
            prefix = full[:target_pos]
            if not prefix:
                continue
            window = prefix[-context:]
            tokens = torch.tensor(window, dtype=torch.long, device=device)
            logits = model.forward(tokens, len(window) - 1)
            target = torch.tensor([full[target_pos]], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target, reduction="sum")
            pred = int(torch.argmax(logits).item())
            total_loss += float(loss.item())
            total_correct += int(pred == full[target_pos])
            total_tokens += 1
            example_tokens += 1
        if example_tokens > 0:
            total_examples += 1
    if total_tokens == 0:
        return EditPolicyMetric(examples=0, response_tokens=0, loss=0.0, accuracy=0.0)
    return EditPolicyMetric(
        examples=total_examples,
        response_tokens=total_tokens,
        loss=total_loss / total_tokens,
        accuracy=total_correct / total_tokens,
    )


def main() -> int:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    cfg = resolve_config(args.checkpoint, args.config_json)
    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    load_lean_checkpoint(args.checkpoint, model)
    model.eval()
    examples = iter_examples(args.dataset, args.split, args.limit)
    if not examples:
        raise SystemExit("no edit-policy examples matched the requested split")
    metric = evaluate_edit_examples(model, examples, context=args.context, device=device)
    print(f"checkpoint={args.checkpoint}")
    print(f"config={cfg.to_dict()}")
    print(f"dataset={args.dataset}")
    print(f"split={args.split}")
    print(f"context={args.context}")
    print(f"examples={metric.examples}")
    print(f"response_tokens={metric.response_tokens}")
    print(f"loss={metric.loss:.6f}")
    print(f"accuracy={metric.accuracy:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
