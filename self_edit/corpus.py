from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CURATED_MATHLIB_STYLE_SNIPPETS = [
    "example : (1 : Nat) = 1 := by decide",
    "example (n : Nat) : n = n := by rfl",
    "example (a b : Nat) : a + b = b + a := by omega",
    "example (xs : List Nat) : xs.length = xs.reverse.reverse.length := by simp",
    "theorem add_zero_right (n : Nat) : n + 0 = n := by simp",
]

CURATED_LEAN_PROGRAM_SNIPPETS = [
    "structure Point where\n  x : Float\n  y : Float",
    "def swapPair (p : Nat × Nat) : Nat × Nat :=\n  (p.2, p.1)",
    "inductive Flag where\n  | on\n  | off",
    "instance : Inhabited Flag where\n  default := Flag.off",
    "def mapOption (f : Nat → Nat) : Option Nat → Option Nat\n  | none => none\n  | some n => some (f n)",
]


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    source_id: str
    file_path: str
    text: str


def load_self_source_document(repo_root: str | Path) -> CorpusDocument:
    root = Path(repo_root)
    source_path = root / "MicroGPT.lean"
    return CorpusDocument(
        source_id="self:MicroGPT.lean",
        file_path=str(source_path),
        text=source_path.read_text(encoding="utf-8"),
    )


def load_curated_documents() -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for idx, text in enumerate(CURATED_MATHLIB_STYLE_SNIPPETS):
        documents.append(
            CorpusDocument(
                source_id=f"curated:mathlib:{idx}",
                file_path=f"curated/mathlib_style_{idx}.lean",
                text=text,
            )
        )
    for idx, text in enumerate(CURATED_LEAN_PROGRAM_SNIPPETS):
        documents.append(
            CorpusDocument(
                source_id=f"curated:program:{idx}",
                file_path=f"curated/program_{idx}.lean",
                text=text,
            )
        )
    return documents


def build_edit_corpus(repo_root: str | Path) -> list[CorpusDocument]:
    return [load_self_source_document(repo_root)] + load_curated_documents()
