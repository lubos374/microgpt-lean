from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.benchmark import BENCHMARK_PROMPTS
from microgpt_torch.checkpoint import load_lean_checkpoint
from microgpt_torch.config import GPTConfigTorch, load_config_json
from microgpt_torch.generate import generate_text
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.train import byte_config


def resolve_config(checkpoint: Path, config_json: Path | None) -> GPTConfigTorch:
    if config_json is not None:
        return load_config_json(config_json)
    inferred = checkpoint.parent / "config.json"
    if inferred.exists():
        return load_config_json(inferred)
    return byte_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate byte-level text from a Lean-compatible MicroGPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint-dir/config.json if present.")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt to continue. Repeat for multiple prompts.")
    parser.add_argument(
        "--benchmark-prompts",
        action="store_true",
        help="Use the fixed Lean benchmark prompt set if --prompt is not provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    prompts = args.prompts or (BENCHMARK_PROMPTS if args.benchmark_prompts else ["def ", "theorem ", "by "])
    cfg = resolve_config(args.checkpoint, args.config_json)

    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    load_lean_checkpoint(args.checkpoint, model)
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    print(f"checkpoint={args.checkpoint}")
    print(f"config={cfg.to_dict()}")
    print(f"context={args.context}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"temperature={args.temperature}")
    print(f"top_k={args.top_k}")
    print(f"greedy={args.greedy}")
    for prompt in prompts:
        token_ids, text = generate_text(
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
        print(f"generated_token_count={len(token_ids)}")
        print(f"generated_text={text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
