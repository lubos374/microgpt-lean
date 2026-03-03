from __future__ import annotations

from .schemas import EditTarget


def build_edit_prompt(
    target: EditTarget,
    *,
    local_context: str | None = None,
) -> str:
    context_block = ""
    if local_context is not None and local_context.strip():
        context_block = f"Local context:\n{local_context}\n\n"
    return (
        "You are editing one Lean declaration body.\n"
        "Produce a replacement body only.\n"
        "Do not repeat the header.\n"
        "Return plain Lean code for the body only.\n\n"
        f"File: {target.file_path}\n"
        f"Declaration: {target.decl_name}\n"
        f"Kind: {target.decl_kind}\n"
        f"Lines: {target.start_line}-{target.end_line}\n\n"
        f"{context_block}"
        "Header:\n"
        f"{target.header_text}\n\n"
        "Current body:\n"
        f"{target.body_text}\n\n"
        f"Replacement body for {target.decl_name}:\n"
    )
