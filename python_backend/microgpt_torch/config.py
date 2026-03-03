from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GPTConfigTorch:
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    n_layers: int
    vocab: int

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError(
                f"expected d_model == n_heads * d_head, got {self.d_model=} {self.n_heads=} {self.d_head=}"
            )
        if self.vocab <= 0:
            raise ValueError(f"expected vocab > 0, got {self.vocab}")
        if self.n_heads <= 0:
            raise ValueError(f"expected n_heads > 0, got {self.n_heads}")
        if self.d_head <= 0:
            raise ValueError(f"expected d_head > 0, got {self.d_head}")
        if self.d_model < 0 or self.d_ff < 0 or self.n_layers < 0:
            raise ValueError("model dimensions must be non-negative")

    @property
    def embedding_param_count(self) -> int:
        return self.d_model * self.vocab

    @property
    def output_head_param_count(self) -> int:
        return 2 * self.d_model + self.vocab * self.d_model

    @property
    def attention_head_param_count(self) -> int:
        return 3 * self.d_head * self.d_model

    @property
    def transformer_layer_param_count(self) -> int:
        head_params = self.n_heads * self.attention_head_param_count
        wo_params = self.d_model * (self.n_heads * self.d_head)
        ln_params = 4 * self.d_model
        mlp_params = self.d_ff * self.d_model + self.d_model * self.d_ff
        return ln_params + head_params + wo_params + mlp_params

    @property
    def param_count(self) -> int:
        return (
            self.embedding_param_count
            + self.n_layers * self.transformer_layer_param_count
            + self.output_head_param_count
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "GPTConfigTorch":
        return cls(
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            d_head=int(data["d_head"]),
            d_ff=int(data["d_ff"]),
            n_layers=int(data["n_layers"]),
            vocab=int(data["vocab"]),
        )


def save_config_json(path: str | Path, cfg: GPTConfigTorch) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config_json(path: str | Path) -> GPTConfigTorch:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return GPTConfigTorch.from_dict(data)
