from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
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


LEAN_MARKERS = [
    "def ",
    "theorem ",
    "by ",
    ":=",
    "where",
    "structure",
    "inductive",
    "instance",
    "List",
    "Nat",
    "Float",
    "Vec",
    "Mat",
    "layer",
    "model",
    "d_model",
    "token",
]


@dataclass(frozen=True, slots=True)
class SampleScore:
    text: str
    printable_ratio: float
    non_space_ratio: float
    marker_hits: int
    newline_count: int
    punctuation_count: int
    score: float


def score_text(text: str) -> SampleScore:
    if not text:
        return SampleScore(text=text, printable_ratio=0.0, non_space_ratio=0.0, marker_hits=0, newline_count=0, punctuation_count=0, score=0.0)
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    non_space = sum(1 for ch in text if not ch.isspace())
    marker_hits = sum(text.count(marker) for marker in LEAN_MARKERS)
    newline_count = text.count("\n")
    punctuation_count = sum(1 for ch in text if ch in ":=()[]{}|->")
    printable_ratio = printable / len(text)
    non_space_ratio = non_space / len(text)
    score = (
        2.0 * marker_hits
        + 0.5 * newline_count
        + 0.05 * punctuation_count
        + 10.0 * printable_ratio
        + 5.0 * min(non_space_ratio, 0.85)
    )
    return SampleScore(
        text=text,
        printable_ratio=printable_ratio,
        non_space_ratio=non_space_ratio,
        marker_hits=marker_hits,
        newline_count=newline_count,
        punctuation_count=punctuation_count,
        score=score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristically score generated Lean-like samples across checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint-dir/config.json if present.")
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt to sample. Repeat for multiple prompts.")
    parser.add_argument(
        "--benchmark-prompts",
        action="store_true",
        help="Use the fixed Lean benchmark prompt set if --prompt is not provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_config(checkpoint_dir: Path, config_json: Path | None) -> GPTConfigTorch:
    if config_json is not None:
        return load_config_json(config_json)
    inferred = checkpoint_dir / "config.json"
    if inferred.exists():
        return load_config_json(inferred)
    return byte_config()


def main() -> int:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    prompts = args.prompts or (BENCHMARK_PROMPTS if args.benchmark_prompts else ["def ", "theorem ", "by "])
    checkpoints = sorted(args.checkpoint_dir.glob(args.glob))
    if args.limit is not None and args.limit > 0:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit(f"no checkpoints matched {args.glob!r} in {args.checkpoint_dir}")

    cfg = resolve_config(args.checkpoint_dir, args.config_json)
    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    for checkpoint in checkpoints:
        load_lean_checkpoint(checkpoint, model)
        model.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

        scores: list[SampleScore] = []
        print(f"checkpoint={checkpoint}")
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
            sample_score = score_text(text)
            scores.append(sample_score)
            print(
                "prompt={prompt!r} score={score:.3f} marker_hits={marker_hits} newlines={newlines} "
                "non_space_ratio={non_space_ratio:.3f}".format(
                    prompt=prompt,
                    score=sample_score.score,
                    marker_hits=sample_score.marker_hits,
                    newlines=sample_score.newline_count,
                    non_space_ratio=sample_score.non_space_ratio,
                )
            )
            print(f"sample={text!r}")
        mean_score = statistics.fmean(score.score for score in scores)
        print(f"mean_score={mean_score:.3f}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
