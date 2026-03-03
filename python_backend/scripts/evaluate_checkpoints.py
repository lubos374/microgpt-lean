from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.checkpoint import load_lean_checkpoint
from microgpt_torch.config import GPTConfigTorch, load_config_json
from microgpt_torch.generate import generate_text
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.parity import LeanParityBridge, run_parity_case
from microgpt_torch.train import byte_config


def resolve_config(checkpoint_dir: Path, config_json: Path | None) -> GPTConfigTorch:
    if config_json is not None:
        return load_config_json(config_json)
    inferred = checkpoint_dir / "config.json"
    if inferred.exists():
        return load_config_json(inferred)
    return byte_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one or more Lean-format byte-model checkpoints with sampling and optional parity.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint-dir/config.json if present.")
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt to sample. Repeat for multiple prompts.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N matching checkpoints.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--lean-cmd", default="lean")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    prompts = args.prompts or ["def ", "theorem ", "by "]
    checkpoints = sorted(args.checkpoint_dir.glob(args.glob))
    if args.limit is not None and args.limit > 0:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit(f"no checkpoints matched {args.glob!r} in {args.checkpoint_dir}")

    cfg = resolve_config(args.checkpoint_dir, args.config_json)
    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    bridge = LeanParityBridge(repo_root=REPO_ROOT, lean_cmd=args.lean_cmd) if args.parity else None

    for checkpoint in checkpoints:
        load_lean_checkpoint(checkpoint, model)
        model.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

        print(f"checkpoint={checkpoint}")
        if bridge is not None:
            default_cfg = GPTConfigTorch(d_model=16, n_heads=1, d_head=16, d_ff=32, n_layers=1, vocab=256)
            if cfg != default_cfg:
                raise SystemExit("--parity is only supported for the default Lean byte fixture config")
            parity = run_parity_case("byte", bridge=bridge, checkpoint_path=checkpoint)
            print(f"parity_within_tolerance={parity.within_tolerance}")
            print(f"parity_max_abs_error={parity.max_abs_error}")
            print(f"parity_loss_abs_error={parity.loss_abs_error}")
        for prompt in prompts:
            _, text = generate_text(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                context=args.context,
                temperature=args.temperature,
                top_k=args.top_k,
                greedy=args.greedy,
                device=device,
                generator=generator,
            )
            print(f"prompt={prompt!r}")
            print(f"sample={text!r}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
