from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microgpt_torch.config import GPTConfigTorch, load_config_json
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.parity import (
    DEFAULT_FIXTURE_CASES,
    FIXTURE_CONFIGS,
    LeanParityBridge,
    run_config_parity_case,
    run_parity_case,
)


def build_config(args: argparse.Namespace) -> GPTConfigTorch | None:
    if args.config_json is not None:
        return load_config_json(args.config_json)
    if all(value is not None for value in (args.d_model, args.n_heads, args.d_head, args.d_ff, args.n_layers, args.vocab)):
        return GPTConfigTorch(
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_head=args.d_head,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            vocab=args.vocab,
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch logits/loss against Lean for one fixture case.")
    parser.add_argument("--fixture", choices=sorted(FIXTURE_CONFIGS))
    parser.add_argument("--config-json", type=Path, help="Use an arbitrary model config from JSON instead of a built-in fixture.")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--d-head", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--checkpoint", type=Path, help="Lean-format checkpoint to load before the parity run.")
    parser.add_argument("--tokens", help="Comma-separated token ids. Defaults to the fixture's built-in case.")
    parser.add_argument("--predict-idx", type=int, help="Prediction index inside the token sequence.")
    parser.add_argument("--target", type=int, help="Target token id for cross-entropy.")
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--lean-cmd", default="lean")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge = LeanParityBridge(repo_root=REPO_ROOT, lean_cmd=args.lean_cmd)
    cfg = build_config(args)
    if cfg is None:
        fixture = args.fixture or "tiny-single-head"
        case = DEFAULT_FIXTURE_CASES[fixture]
        tokens = case.tokens if args.tokens is None else [int(part) for part in args.tokens.split(",") if part]
        predict_idx = case.predict_idx if args.predict_idx is None else args.predict_idx
        target = case.target if args.target is None else args.target
        model = MicroGPTTorch(FIXTURE_CONFIGS[fixture])
        result = run_parity_case(
            fixture,
            bridge=bridge,
            model=model,
            checkpoint_path=args.checkpoint,
            tokens=tokens,
            predict_idx=predict_idx,
            target=target,
            atol=args.atol,
            rtol=args.rtol,
        )
    else:
        if args.tokens is None or args.predict_idx is None or args.target is None:
            raise SystemExit("custom-config parity requires --tokens, --predict-idx, and --target")
        tokens = [int(part) for part in args.tokens.split(",") if part]
        result = run_config_parity_case(
            cfg,
            bridge=bridge,
            checkpoint_path=args.checkpoint,
            tokens=tokens,
            predict_idx=args.predict_idx,
            target=args.target,
            atol=args.atol,
            rtol=args.rtol,
        )

    print(f"fixture={result.fixture}")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"lean_logits={result.lean_logits}")
    print(f"torch_logits={result.torch_logits}")
    print(f"lean_loss={result.lean_loss}")
    print(f"torch_loss={result.torch_loss}")
    print(f"max_abs_error={result.max_abs_error}")
    print(f"max_rel_error={result.max_rel_error}")
    print(f"loss_abs_error={result.loss_abs_error}")
    print(f"logits_within_tolerance={result.logits_within_tolerance}")
    print(f"loss_within_tolerance={result.loss_within_tolerance}")
    print(f"within_tolerance={result.within_tolerance}")
    return 0 if result.within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
