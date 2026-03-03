from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class EditTarget:
    file_path: str
    decl_name: str
    decl_kind: Literal["def", "theorem", "lemma"]
    start_line: int
    end_line: int
    header_text: str
    body_text: str

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EditPromptExample:
    split: Literal["train", "valid"]
    source_id: str
    prompt: str
    response: str
    target: EditTarget

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_id": self.source_id,
            "prompt": self.prompt,
            "response": self.response,
            "target": self.target.to_dict(),
        }
