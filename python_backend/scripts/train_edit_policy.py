from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.checkpoint import load_lean_checkpoint
from microgpt_torch.config import GPTConfigTorch, load_config_json, save_config_json
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.train import (
    TokenExample,
    encode_utf8_bytes,
    read_training_state_metadata,
    train_byte_model,
)


def infer_start_step_from_checkpoint(path: Path) -> int:
    match = re.search(r"step-(\d+)\.bin$", path.name)
    if match is not None:
        return int(match.group(1))
    return 0


def infer_optimizer_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".optim.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the current byte model on declaration edit prompt/response pairs.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Starting Lean-format checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL edit dataset from build_edit_dataset.py")
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint.parent/config.json if present.")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-scale", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-optim-from", type=Path, help="Resume optimizer state from a .optim.pt sidecar.")
    parser.add_argument("--start-step", type=int, help="Override the inferred initial global step.")
    parser.add_argument("--train-eval-limit", type=int, default=2048)
    parser.add_argument("--valid-eval-limit", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_config(checkpoint: Path, config_json: Path | None) -> GPTConfigTorch:
    if config_json is not None:
        return load_config_json(config_json)
    inferred = checkpoint.parent / "config.json"
    if inferred.exists():
        return load_config_json(inferred)
    raise SystemExit("config.json is required for non-default edit-policy fine-tuning")


def iter_dataset_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def windows_from_prompt_response(prompt: str, response: str, *, context: int) -> list[TokenExample]:
    prompt_ids = encode_utf8_bytes(prompt)
    response_ids = encode_utf8_bytes(response)
    full = prompt_ids + response_ids
    examples: list[TokenExample] = []
    if context <= 0:
        return examples
    for target_index in range(len(prompt_ids), len(full)):
        prefix = full[:target_index]
        if len(prefix) < context:
            continue
        examples.append(
            TokenExample(
                tokens=prefix[-context:],
                predict_idx=context - 1,
                target=full[target_index],
            )
        )
    return examples


def build_window_dataset(path: Path, *, split: str, context: int) -> list[TokenExample]:
    examples: list[TokenExample] = []
    for row in iter_dataset_rows(path):
        if split != "all" and row["split"] != split:
            continue
        examples.extend(
            windows_from_prompt_response(
                row["prompt"],
                row["response"],
                context=context,
            )
        )
    return examples


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    cfg = resolve_config(args.checkpoint, args.config_json)
    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    load_lean_checkpoint(args.checkpoint, model)

    train_examples = build_window_dataset(args.dataset, split="train", context=args.context)
    valid_examples = build_window_dataset(args.dataset, split="valid", context=args.context)
    if not train_examples:
        raise SystemExit("no train windows were generated from the edit dataset")
    if not valid_examples:
        raise SystemExit("no valid windows were generated from the edit dataset")

    train_batches = math.ceil(len(train_examples) / args.batch_size)
    steps = max(args.epochs * train_batches, 1)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(args.checkpoint_dir / "config.json", cfg)

    start_step = infer_start_step_from_checkpoint(args.checkpoint)
    resume_optimizer_state = args.resume_optim_from
    if resume_optimizer_state is None:
        inferred = infer_optimizer_state_path(args.checkpoint)
        if inferred.exists():
            resume_optimizer_state = inferred
    if start_step == 0 and resume_optimizer_state is not None and resume_optimizer_state.exists():
        restored_step, _ = read_training_state_metadata(resume_optimizer_state, device="cpu")
        if restored_step is not None:
            start_step = restored_step
    if args.start_step is not None:
        start_step = args.start_step

    logs = train_byte_model(
        model,
        train_examples,
        valid_examples,
        start_step=start_step,
        steps=steps,
        lr=args.lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        train_eval_limit=args.train_eval_limit,
        valid_eval_limit=args.valid_eval_limit,
        save_best_on_valid=True,
        grad_clip_norm=args.grad_clip_norm,
        warmup_steps=args.warmup_steps,
        min_lr_scale=args.min_lr_scale,
        shuffle_train=True,
        seed=args.seed,
        resume_optimizer_state_path=resume_optimizer_state,
        save_optimizer_state=True,
        device=device,
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"config={cfg.to_dict()}")
    print(f"dataset={args.dataset}")
    print(f"context={args.context}")
    print(f"epochs={args.epochs}")
    print(f"steps={steps}")
    print(f"start_step={start_step}")
    print(f"train_windows={len(train_examples)}")
    print(f"valid_windows={len(valid_examples)}")
    for log in logs:
        print(
            "step={step} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            "lr={lr:.8f} grad_norm={grad_norm:.6f} tokens_per_second={tps:.2f}".format(
                step=log.step,
                train_loss=log.train_loss,
                valid_loss=log.valid_loss,
                lr=log.lr,
                grad_norm=log.grad_norm,
                tps=log.tokens_per_second,
            )
        )
    print(f"latest_checkpoint={args.checkpoint_dir / 'latest.bin'}")
    print(f"best_checkpoint={args.checkpoint_dir / 'best.bin'}")
    print(f"latest_optimizer_state={args.checkpoint_dir / 'latest.optim.pt'}")
    print(f"train_log_jsonl={args.checkpoint_dir / 'train_log.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
