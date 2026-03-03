from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.checkpoint import load_lean_checkpoint
from microgpt_torch.config import GPTConfigTorch, load_config_json
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.train import (
    CURATED_LEAN_PROGRAM_SNIPPETS,
    CURATED_MATHLIB_STYLE_SNIPPETS,
    TokenExample,
    byte_config,
    byte_token_examples_from_texts,
    load_priority_training_texts,
    split_byte_token_examples_from_texts,
    token_examples_to_tensors,
)


@dataclass(frozen=True, slots=True)
class CheckpointMetric:
    checkpoint: Path
    loss: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure exact next-byte loss and top-1 accuracy for one or more checkpoints.")
    parser.add_argument("--checkpoint", type=Path, help="Single checkpoint to evaluate.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory of checkpoints to evaluate.")
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint-dir/config.json if present.")
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--dataset", choices=["priority-valid", "priority-train", "curated"], default="priority-valid")
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", choices=["accuracy", "loss"], default="accuracy")
    parser.add_argument("--select-best-to", type=Path, help="Copy the best checkpoint by --metric to this path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def build_dataset(kind: str, context: int, holdout_every: int, limit: int | None) -> Sequence[TokenExample]:
    if kind == "curated":
        texts = CURATED_MATHLIB_STYLE_SNIPPETS + CURATED_LEAN_PROGRAM_SNIPPETS
        examples = byte_token_examples_from_texts(context, texts)
    else:
        texts = load_priority_training_texts(REPO_ROOT)
        split = split_byte_token_examples_from_texts(context, holdout_every, texts)
        examples = split.valid if kind == "priority-valid" else split.train
    if limit is not None and limit > 0:
        return examples[:limit]
    return examples


@torch.no_grad()
def evaluate_dataset(
    model: MicroGPTTorch,
    dataset: Sequence[TokenExample],
    *,
    batch_size: int,
    device: torch.device | str | None = None,
) -> tuple[float, float]:
    if not dataset:
        return 0.0, 0.0
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        tokens, predict_idx, target = token_examples_to_tensors(batch, device=device)
        logits = model.forward_batch(tokens, predict_idx)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
        pred = torch.argmax(logits, dim=-1)
        total_loss += float(loss.item())
        total_correct += int((pred == target).sum().item())
        total_examples += len(batch)
    return total_loss / total_examples, total_correct / total_examples


def iter_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint is not None:
        return [args.checkpoint]
    if args.checkpoint_dir is not None:
        checkpoints = sorted(args.checkpoint_dir.glob(args.glob))
        if checkpoints:
            return checkpoints
        raise SystemExit(f"no checkpoints matched {args.glob!r} in {args.checkpoint_dir}")
    raise SystemExit("provide either --checkpoint or --checkpoint-dir")


def resolve_config(args: argparse.Namespace) -> GPTConfigTorch:
    if args.config_json is not None:
        return load_config_json(args.config_json)
    if args.checkpoint is not None:
        inferred = args.checkpoint.parent / "config.json"
        if inferred.exists():
            return load_config_json(inferred)
    if args.checkpoint_dir is not None:
        inferred = args.checkpoint_dir / "config.json"
        if inferred.exists():
            return load_config_json(inferred)
    return byte_config()


def main() -> int:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    dataset = build_dataset(args.dataset, args.context, args.holdout_every, args.limit)
    if not dataset:
        raise SystemExit("evaluation dataset is empty")

    cfg = resolve_config(args)
    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    results: list[CheckpointMetric] = []
    for checkpoint in iter_checkpoints(args):
        load_lean_checkpoint(checkpoint, model)
        model.eval()
        loss, accuracy = evaluate_dataset(model, dataset, batch_size=args.batch_size, device=device)
        results.append(CheckpointMetric(checkpoint=checkpoint, loss=loss, accuracy=accuracy))
        print(f"checkpoint={checkpoint}")
        print(f"config={cfg.to_dict()}")
        print(f"dataset={args.dataset}")
        print(f"context={args.context}")
        print(f"examples={len(dataset)}")
        print(f"loss={loss:.6f}")
        print(f"accuracy={accuracy:.6f}")
        print("")

    if results:
        if args.metric == "accuracy":
            best = max(results, key=lambda item: (item.accuracy, -item.loss))
        else:
            best = min(results, key=lambda item: (item.loss, -item.accuracy))
        print(f"best_metric={args.metric}")
        print(f"best_checkpoint={best.checkpoint}")
        print(f"best_loss={best.loss:.6f}")
        print(f"best_accuracy={best.accuracy:.6f}")
        if args.select_best_to is not None:
            args.select_best_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best.checkpoint, args.select_best_to)
            sidecar = best.checkpoint.parent / "config.json"
            if sidecar.exists():
                dest_config = args.select_best_to.parent / "config.json"
                if sidecar.resolve() != dest_config.resolve():
                    shutil.copyfile(sidecar, dest_config)
            summary_path = args.select_best_to.with_suffix(".txt")
            summary_path.write_text(
                "metric={metric}\ncheckpoint={checkpoint}\nloss={loss:.9f}\naccuracy={accuracy:.9f}\n".format(
                    metric=args.metric,
                    checkpoint=best.checkpoint,
                    loss=best.loss,
                    accuracy=best.accuracy,
                ),
                encoding="utf-8",
            )
            print(f"selected_checkpoint={args.select_best_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
