from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.checkpoint import load_lean_checkpoint
from microgpt_torch.config import GPTConfigTorch, load_config_json, save_config_json
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.parity import LeanParityBridge, run_parity_case
from microgpt_torch.train import (
    read_training_state_metadata,
    load_priority_training_texts,
    split_byte_token_examples_from_texts,
    train_byte_model,
)
from microgpt_torch.generate import generate_text


def infer_start_step_from_checkpoint(path: Path) -> int:
    match = re.search(r"step-(\d+)\.bin$", path.name)
    if match is not None:
        return int(match.group(1))
    return 0


def infer_config_json_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "config.json"


def infer_optimizer_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".optim.pt")


def build_model_config(args: argparse.Namespace) -> GPTConfigTorch:
    if args.config_json is not None:
        return load_config_json(args.config_json)
    if args.resume_from is not None:
        config_path = infer_config_json_path(args.resume_from)
        if config_path.exists():
            return load_config_json(config_path)
    return GPTConfigTorch(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_head,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        vocab=args.vocab,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the byte-level PyTorch mirror and export Lean-compatible checkpoints.")
    parser.add_argument("--config-json", type=Path, help="Load model config from JSON instead of CLI dimension flags.")
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, help="Clip gradient norm to this value before each optimizer step.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "python_backend" / "checkpoints")
    parser.add_argument("--resume-from", type=Path, help="Resume model weights from a Lean-format checkpoint.")
    parser.add_argument("--resume-optim-from", type=Path, help="Resume PyTorch optimizer state from a .optim.pt sidecar.")
    parser.add_argument("--start-step", type=int, help="Override the initial global step number for resumed training.")
    parser.add_argument("--train-eval-limit", type=int, help="Evaluate train loss on only the first N examples at each log step.")
    parser.add_argument("--valid-eval-limit", type=int, help="Evaluate validation loss on only the first N examples at each log step.")
    parser.add_argument(
        "--best-on-valid-loss",
        action="store_true",
        help="Save checkpoint-dir/best.bin whenever the logged validation loss improves.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable train-set reshuffling between epochs.")
    parser.add_argument("--no-optimizer-state", action="store_true", help="Do not write PyTorch optimizer sidecars next to checkpoints.")
    parser.add_argument("--lean-cmd", default="lean")
    parser.add_argument(
        "--sample-after",
        action="store_true",
        help="Generate from the trained model after the run using a few byte-level prompts.",
    )
    parser.add_argument(
        "--sample-prompt",
        action="append",
        dest="sample_prompts",
        help="Prompt used when --sample-after is enabled. Repeat for multiple prompts.",
    )
    parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-k", type=int)
    parser.add_argument("--sample-greedy", action="store_true")
    parser.add_argument(
        "--parity-smoke",
        action="store_true",
        help="After training, load the latest checkpoint in Lean and compare one byte fixture case.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    cfg = build_model_config(args)

    texts = load_priority_training_texts(REPO_ROOT)
    split = split_byte_token_examples_from_texts(args.context, args.holdout_every, texts)
    if not split.train:
        raise SystemExit("train split is empty")

    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    start_step = 0
    resume_optimizer_state = args.resume_optim_from
    if args.resume_from is not None:
        load_lean_checkpoint(args.resume_from, model)
        start_step = infer_start_step_from_checkpoint(args.resume_from)
        if resume_optimizer_state is None:
            inferred_optim = infer_optimizer_state_path(args.resume_from)
            if inferred_optim.exists():
                resume_optimizer_state = inferred_optim
        if start_step == 0 and resume_optimizer_state is not None and resume_optimizer_state.exists():
            restored_step, _ = read_training_state_metadata(resume_optimizer_state, device="cpu")
            if restored_step is not None:
                start_step = restored_step
    if args.start_step is not None:
        start_step = args.start_step
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(args.checkpoint_dir / "config.json", cfg)
    logs = train_byte_model(
        model,
        split.train,
        split.valid,
        start_step=start_step,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        train_eval_limit=args.train_eval_limit,
        valid_eval_limit=args.valid_eval_limit,
        save_best_on_valid=args.best_on_valid_loss,
        grad_clip_norm=args.grad_clip_norm,
        warmup_steps=args.warmup_steps,
        min_lr_scale=args.min_lr_scale,
        shuffle_train=not args.no_shuffle,
        seed=args.seed,
        resume_optimizer_state_path=resume_optimizer_state,
        save_optimizer_state=not args.no_optimizer_state,
        device=device,
    )

    print(f"train_examples={len(split.train)}")
    print(f"valid_examples={len(split.valid)}")
    print(f"config={cfg.to_dict()}")
    print(f"start_step={start_step}")
    for log in logs:
        print(
            "step={step} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            "train_eval_examples={train_eval_examples} valid_eval_examples={valid_eval_examples} "
            "lr={lr:.8f} grad_norm={grad_norm:.6f} tokens_per_second={tps:.2f}".format(
                step=log.step,
                train_loss=log.train_loss,
                valid_loss=log.valid_loss,
                train_eval_examples=log.train_eval_examples,
                valid_eval_examples=log.valid_eval_examples,
                lr=log.lr,
                grad_norm=log.grad_norm,
                tps=log.tokens_per_second,
            )
        )

    latest_checkpoint = args.checkpoint_dir / "latest.bin"
    print(f"latest_checkpoint={latest_checkpoint}")
    if not args.no_optimizer_state:
        print(f"latest_optimizer_state={args.checkpoint_dir / 'latest.optim.pt'}")
    print(f"train_log_jsonl={args.checkpoint_dir / 'train_log.jsonl'}")
    if logs:
        best_log = min(logs, key=lambda log: log.valid_loss)
        print(f"best_logged_step={best_log.step}")
        print(f"best_logged_valid_loss={best_log.valid_loss:.6f}")
        if args.best_on_valid_loss:
            print(f"best_checkpoint={args.checkpoint_dir / 'best.bin'}")

    if args.parity_smoke:
        expected_default = GPTConfigTorch(d_model=16, n_heads=1, d_head=16, d_ff=32, n_layers=1, vocab=256)
        if cfg != expected_default:
            raise SystemExit("--parity-smoke is only supported for the default Lean byte fixture config")
        bridge = LeanParityBridge(repo_root=REPO_ROOT, lean_cmd=args.lean_cmd)
        parity = run_parity_case("byte", bridge=bridge, checkpoint_path=latest_checkpoint)
        print(f"parity_max_abs_error={parity.max_abs_error}")
        print(f"parity_loss_abs_error={parity.loss_abs_error}")
        print(f"parity_within_tolerance={parity.within_tolerance}")

    if args.sample_after:
        prompts = args.sample_prompts or ["def ", "theorem ", "by "]
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        for prompt in prompts:
            _, text = generate_text(
                model,
                prompt,
                max_new_tokens=args.sample_max_new_tokens,
                context=args.context,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
                greedy=args.sample_greedy,
                device=device,
                generator=generator,
            )
            print(f"sample_prompt={prompt!r}")
            print(f"sample_text={text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
