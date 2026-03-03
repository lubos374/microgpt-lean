from __future__ import annotations

import argparse
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
from microgpt_torch.config import GPTConfigTorch, load_config_json
from microgpt_torch.generate import generate_text
from microgpt_torch.model import MicroGPTTorch
from microgpt_torch.train import byte_config
from self_edit.corpus import build_edit_corpus
from self_edit.declarations import extract_edit_targets, surrounding_context
from self_edit.prompts import build_edit_prompt
from self_edit.schemas import EditTarget

DECL_HEADER_RE = re.compile(r"(?m)^\s*(def|theorem|lemma|structure|inductive|class|instance)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a replacement body proposal for one extracted Lean declaration.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decl-name", required=True, help="Exact declaration name, e.g. Vec.ofFn")
    parser.add_argument("--config-json", type=Path, help="Model config JSON. Defaults to checkpoint.parent/config.json if present.")
    parser.add_argument("--file-path", help="Optional file path filter when a declaration name is ambiguous.")
    parser.add_argument("--max-lines", type=int, default=80)
    parser.add_argument("--context-before", type=int, default=4)
    parser.add_argument("--context-after", type=int, default=4)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--filter-invalid", action="store_true")
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
    return byte_config()


def select_target(
    decl_name: str,
    *,
    file_path: str | None,
    max_lines: int,
) -> tuple[EditTarget, str]:
    for document in build_edit_corpus(REPO_ROOT):
        if file_path is not None and document.file_path != file_path:
            continue
        targets = extract_edit_targets(document.file_path, document.text, max_lines=max_lines)
        for target in targets:
            if target.decl_name == decl_name:
                return target, document.text
    raise SystemExit(f"declaration not found: {decl_name}")


def validate_proposal_body(body: str) -> list[str]:
    issues: list[str] = []
    stripped = body.strip()
    if not stripped:
        issues.append("empty")
        return issues
    if "```" in body:
        issues.append("contains_fenced_code")
    if "\uFFFD" in body:
        issues.append("contains_invalid_utf8_replacement")
    if DECL_HEADER_RE.search(body):
        issues.append("contains_new_declaration_header")
    return issues


def main() -> int:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    cfg = resolve_config(args.checkpoint, args.config_json)
    model = MicroGPTTorch(cfg, dtype=dtype, device=device)
    load_lean_checkpoint(args.checkpoint, model)
    model.eval()

    target, document_text = select_target(args.decl_name, file_path=args.file_path, max_lines=args.max_lines)
    prompt = build_edit_prompt(
        target,
        local_context=surrounding_context(
            document_text,
            start_line=target.start_line,
            end_line=target.end_line,
            before=args.context_before,
            after=args.context_after,
        ),
    )

    print(f"checkpoint={args.checkpoint}")
    print(f"config={cfg.to_dict()}")
    print(f"decl_name={target.decl_name}")
    print(f"file_path={target.file_path}")
    print(f"lines={target.start_line}-{target.end_line}")
    print("header:")
    print(target.header_text)
    print("current_body:")
    print(target.body_text)
    for sample_idx in range(max(args.num_samples, 1)):
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + sample_idx)
        _, generated = generate_text(
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
        proposal = generated[len(prompt) :]
        issues = validate_proposal_body(proposal)
        if args.filter_invalid and issues:
            continue
        print(f"sample={sample_idx}")
        print(f"valid={not issues}")
        if issues:
            print(f"issues={','.join(issues)}")
        print("proposed_body:")
        print(proposal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
