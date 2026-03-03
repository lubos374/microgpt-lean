from __future__ import annotations

import re
from typing import Iterable

from .schemas import EditTarget


DECL_START_RE = re.compile(r"^(def|theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)\b")


def find_decl_starts(lines: list[str]) -> list[tuple[int, str, str]]:
    starts: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        if line.lstrip() != line:
            continue
        match = DECL_START_RE.match(line)
        if match is None:
            continue
        starts.append((idx, match.group(1), match.group(2)))
    return starts


def split_header_body(block: str) -> tuple[str, str] | None:
    marker_idx = block.find(":=")
    if marker_idx < 0:
        return None
    header = block[: marker_idx + 2]
    body = block[marker_idx + 2 :]
    return header.rstrip(), body.lstrip()


def trim_trailing_comment_suffix(block_lines: list[str]) -> list[str]:
    end = len(block_lines)
    while end > 0 and block_lines[end - 1].strip() == "":
        end -= 1
    while end > 0:
        stripped = block_lines[end - 1].lstrip()
        if stripped.startswith("/--") or stripped.startswith("--") or stripped.startswith("/-") or stripped == "-/":
            end -= 1
            while end > 0 and block_lines[end - 1].strip() == "":
                end -= 1
            continue
        break
    return block_lines[:end]


def extract_edit_targets(
    file_path: str,
    text: str,
    *,
    max_lines: int = 80,
) -> list[EditTarget]:
    lines = text.splitlines()
    starts = find_decl_starts(lines)
    targets: list[EditTarget] = []
    for pos, (start_idx, decl_kind, decl_name) in enumerate(starts):
        end_idx = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        block_lines = trim_trailing_comment_suffix(lines[start_idx:end_idx])
        if not block_lines:
            continue
        if max_lines > 0 and len(block_lines) > max_lines:
            continue
        block = "\n".join(block_lines).strip()
        if "sorry" in block:
            continue
        header_body = split_header_body(block)
        if header_body is None:
            continue
        header_text, body_text = header_body
        if not body_text.strip():
            continue
        targets.append(
            EditTarget(
                file_path=file_path,
                decl_name=decl_name,
                decl_kind=decl_kind,  # type: ignore[arg-type]
                start_line=start_idx + 1,
                end_line=end_idx,
                header_text=header_text,
                body_text=body_text,
            )
        )
    return targets


def surrounding_context(
    text: str,
    *,
    start_line: int,
    end_line: int,
    before: int = 4,
    after: int = 4,
) -> str:
    lines = text.splitlines()
    start_idx = max(start_line - 1 - before, 0)
    end_idx = min(end_line + after, len(lines))
    return "\n".join(lines[start_idx:end_idx])


def count_targets_by_file(targets: Iterable[EditTarget]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.file_path] = counts.get(target.file_path, 0) + 1
    return counts
