from __future__ import annotations

import struct
from pathlib import Path
from typing import Protocol, Sequence


class FlatParamModel(Protocol):
    def to_flat_params(self) -> list[float]: ...

    def load_flat_params(self, values: Sequence[float]) -> None: ...


def read_lean_checkpoint(path: str | Path) -> list[float]:
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError("checkpoint missing parameter-count header")
    (count,) = struct.unpack("<Q", raw[:8])
    expected_size = 8 + count * 8
    if len(raw) != expected_size:
        raise ValueError(
            f"checkpoint payload size mismatch: got {len(raw)} bytes, expected {expected_size}"
        )
    return [value[0] for value in struct.iter_unpack("<d", raw[8:])]


def write_lean_checkpoint(path: str | Path, values: Sequence[float]) -> None:
    values_list = [float(value) for value in values]
    header = struct.pack("<Q", len(values_list))
    payload = b"".join(struct.pack("<d", value) for value in values_list)
    Path(path).write_bytes(header + payload)


def load_lean_checkpoint(path: str | Path, model: FlatParamModel) -> None:
    model.load_flat_params(read_lean_checkpoint(path))


def save_lean_checkpoint(path: str | Path, model: FlatParamModel) -> None:
    write_lean_checkpoint(path, model.to_flat_params())
