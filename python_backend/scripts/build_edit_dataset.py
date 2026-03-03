from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_edit.corpus import build_edit_corpus
from self_edit.declarations import count_targets_by_file, extract_edit_targets, surrounding_context
from self_edit.prompts import build_edit_prompt
from self_edit.schemas import EditPromptExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a declaration-level edit dataset from the Lean training corpus.")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path.")
    parser.add_argument("--summary-json", type=Path, help="Optional summary JSON path. Defaults to output.with_suffix('.summary.json').")
    parser.add_argument("--max-lines", type=int, default=80)
    parser.add_argument("--context-before", type=int, default=4)
    parser.add_argument("--context-after", type=int, default=4)
    parser.add_argument("--holdout-every", type=int, default=5, help="Assign every Nth source document to validation.")
    return parser.parse_args()


def split_name(index: int, holdout_every: int) -> str:
    if holdout_every > 0 and index % holdout_every == 0:
        return "valid"
    return "train"


def split_names_for_targets(doc_index: int, num_targets: int, holdout_every: int) -> list[str]:
    if num_targets <= 0:
        return []
    if holdout_every <= 0:
        return ["train"] * num_targets
    if num_targets >= holdout_every:
        return [
            "valid" if (doc_index + target_index) % holdout_every == 0 else "train"
            for target_index in range(num_targets)
        ]
    split = split_name(doc_index, holdout_every)
    return [split] * num_targets


def main() -> int:
    args = parse_args()
    documents = build_edit_corpus(REPO_ROOT)
    examples: list[EditPromptExample] = []
    split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    all_targets = []

    for doc_index, document in enumerate(documents):
        targets = extract_edit_targets(document.file_path, document.text, max_lines=args.max_lines)
        all_targets.extend(targets)
        target_splits = split_names_for_targets(doc_index, len(targets), args.holdout_every)
        for target, split in zip(targets, target_splits):
            prompt = build_edit_prompt(
                target,
                local_context=surrounding_context(
                    document.text,
                    start_line=target.start_line,
                    end_line=target.end_line,
                    before=args.context_before,
                    after=args.context_after,
                ),
            )
            response = target.body_text.strip()
            if not response:
                continue
            examples.append(
                EditPromptExample(
                    split=split,  # type: ignore[arg-type]
                    source_id=document.source_id,
                    prompt=prompt,
                    response=response,
                    target=target,
                )
            )
            split_counts[split] += 1
            kind_counts[target.decl_kind] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=True) + "\n")

    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary = {
        "documents": len(documents),
        "examples": len(examples),
        "split_counts": dict(split_counts),
        "kind_counts": dict(kind_counts),
        "targets_by_file": count_targets_by_file(all_targets),
        "split_strategy": "file-level with deterministic declaration-level fallback for large files",
        "max_lines": args.max_lines,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "holdout_every": args.holdout_every,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"documents={len(documents)}")
    print(f"examples={len(examples)}")
    print(f"train_examples={split_counts.get('train', 0)}")
    print(f"valid_examples={split_counts.get('valid', 0)}")
    print(f"kind_counts={dict(kind_counts)}")
    print(f"output={args.output}")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
