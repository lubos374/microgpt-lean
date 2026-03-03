from __future__ import annotations

import ast
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .checkpoint import load_lean_checkpoint
from .config import GPTConfigTorch
from .model import MicroGPTTorch
from .train import byte_config, default_byte_example, tiny_single_head_config, tiny_two_layer_config


FIXTURE_CONFIGS: dict[str, GPTConfigTorch] = {
    "tiny-single-head": tiny_single_head_config(),
    "tiny-two-layer": tiny_two_layer_config(),
    "byte": byte_config(),
}


@dataclass(frozen=True, slots=True)
class FixtureCase:
    tokens: list[int]
    predict_idx: int
    target: int


DEFAULT_FIXTURE_CASES: dict[str, FixtureCase] = {
    "tiny-single-head": FixtureCase(tokens=[0, 1], predict_idx=1, target=0),
    "tiny-two-layer": FixtureCase(tokens=[0, 1], predict_idx=1, target=0),
    "byte": FixtureCase(
        tokens=default_byte_example().tokens,
        predict_idx=default_byte_example().predict_idx,
        target=default_byte_example().target,
    ),
}


@dataclass(frozen=True, slots=True)
class LeanEvalResult:
    logits: list[float]
    loss: float


@dataclass(frozen=True, slots=True)
class ParityResult:
    fixture: str
    checkpoint_path: str
    lean_logits: list[float]
    torch_logits: list[float]
    lean_loss: float
    torch_loss: float
    max_abs_error: float
    max_rel_error: float
    loss_abs_error: float
    logits_within_tolerance: bool
    loss_within_tolerance: bool

    @property
    def within_tolerance(self) -> bool:
        return self.logits_within_tolerance and self.loss_within_tolerance


class LeanParityBridge:
    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        lean_file: str | Path | None = None,
        lean_cmd: str = "lean",
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.lean_file = Path(lean_file) if lean_file is not None else self.repo_root / "MicroGPT.lean"
        self.lean_cmd = lean_cmd

    def _run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.lean_cmd, "--run", str(self.lean_file), *args],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def save_fixture_checkpoint(self, fixture: str, path: str | Path) -> None:
        self._run(["parity-save", fixture, str(path)])

    def save_config_checkpoint(self, cfg: GPTConfigTorch, path: str | Path) -> None:
        self._run(
            [
                "parity-save-config",
                str(cfg.d_model),
                str(cfg.n_heads),
                str(cfg.d_head),
                str(cfg.d_ff),
                str(cfg.n_layers),
                str(cfg.vocab),
                str(path),
            ]
        )

    @staticmethod
    def _parse_eval_result(stdout: str) -> LeanEvalResult:
        parsed: dict[str, str] = {}
        for line in stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                parsed[key.strip()] = value.strip()
        return LeanEvalResult(
            logits=[float(value) for value in ast.literal_eval(parsed["logits"])],
            loss=float(parsed["loss"]),
        )

    def default_case(self, fixture: str) -> FixtureCase:
        result = self._run(["parity-default-case", fixture])
        parsed: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                parsed[key.strip()] = value.strip()
        return FixtureCase(
            tokens=[int(value) for value in ast.literal_eval(parsed["tokens"])],
            predict_idx=int(parsed["predict_idx"]),
            target=int(parsed["target"]),
        )

    def eval_fixture(
        self,
        fixture: str,
        *,
        tokens: Sequence[int],
        predict_idx: int,
        target: int,
        checkpoint: str | Path | None = None,
    ) -> LeanEvalResult:
        checkpoint_arg = "-" if checkpoint is None else str(checkpoint)
        token_csv = ",".join(str(token) for token in tokens)
        result = self._run(
            [
                "parity-eval",
                fixture,
                checkpoint_arg,
                token_csv,
                str(predict_idx),
                str(target),
            ]
        )
        return self._parse_eval_result(result.stdout)

    def eval_config(
        self,
        cfg: GPTConfigTorch,
        *,
        tokens: Sequence[int],
        predict_idx: int,
        target: int,
        checkpoint: str | Path | None = None,
    ) -> LeanEvalResult:
        checkpoint_arg = "-" if checkpoint is None else str(checkpoint)
        token_csv = ",".join(str(token) for token in tokens)
        result = self._run(
            [
                "parity-eval-config",
                str(cfg.d_model),
                str(cfg.n_heads),
                str(cfg.d_head),
                str(cfg.d_ff),
                str(cfg.n_layers),
                str(cfg.vocab),
                checkpoint_arg,
                token_csv,
                str(predict_idx),
                str(target),
            ]
        )
        return self._parse_eval_result(result.stdout)


def cross_entropy_from_logits(logits: torch.Tensor, target: int) -> float:
    target_tensor = torch.tensor([target], dtype=torch.long, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target_tensor)
    return float(loss.item())


def compare_logits(
    lean_logits: Sequence[float],
    torch_logits: Sequence[float],
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> dict[str, float | bool]:
    if len(lean_logits) != len(torch_logits):
        raise ValueError(
            f"logit length mismatch: lean={len(lean_logits)} torch={len(torch_logits)}"
        )
    max_abs_error = 0.0
    max_rel_error = 0.0
    within = True
    for lean_value, torch_value in zip(lean_logits, torch_logits, strict=True):
        abs_error = abs(lean_value - torch_value)
        denom = max(abs(lean_value), abs(torch_value), 1.0e-12)
        rel_error = abs_error / denom
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        if abs_error > atol + rtol * denom:
            within = False
    return {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "within_tolerance": within,
    }


def run_parity_case(
    fixture: str,
    *,
    bridge: LeanParityBridge | None = None,
    model: MicroGPTTorch | None = None,
    checkpoint_path: str | Path | None = None,
    tokens: Sequence[int] | None = None,
    predict_idx: int | None = None,
    target: int | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> ParityResult:
    if fixture not in FIXTURE_CONFIGS:
        raise ValueError(f"unknown fixture: {fixture}")

    bridge = bridge or LeanParityBridge()
    config = FIXTURE_CONFIGS[fixture]
    case = DEFAULT_FIXTURE_CASES[fixture]
    tokens = list(tokens if tokens is not None else case.tokens)
    predict_idx = case.predict_idx if predict_idx is None else predict_idx
    target = case.target if target is None else target

    checkpoint_owned = False
    if checkpoint_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix=f"{fixture}-", suffix=".bin", delete=False)
        tmp.close()
        checkpoint_path = tmp.name
        bridge.save_fixture_checkpoint(fixture, checkpoint_path)
        checkpoint_owned = True

    if model is None:
        model = MicroGPTTorch(config, dtype=torch.float64)
    load_lean_checkpoint(checkpoint_path, model)
    model.eval()

    lean_eval = bridge.eval_fixture(
        fixture,
        tokens=tokens,
        predict_idx=predict_idx,
        target=target,
        checkpoint=checkpoint_path,
    )
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    with torch.no_grad():
        torch_logits_tensor = model.forward(token_tensor, predict_idx)
    torch_logits = [float(value) for value in torch_logits_tensor.detach().cpu().tolist()]
    torch_loss = cross_entropy_from_logits(torch_logits_tensor, target)

    logit_cmp = compare_logits(lean_eval.logits, torch_logits, atol=atol, rtol=rtol)
    loss_abs_error = abs(lean_eval.loss - torch_loss)
    loss_within_tolerance = loss_abs_error <= atol + rtol * max(
        abs(lean_eval.loss), abs(torch_loss), 1.0e-12
    )

    if checkpoint_owned:
        Path(checkpoint_path).unlink(missing_ok=True)

    return ParityResult(
        fixture=fixture,
        checkpoint_path=str(checkpoint_path),
        lean_logits=lean_eval.logits,
        torch_logits=torch_logits,
        lean_loss=lean_eval.loss,
        torch_loss=torch_loss,
        max_abs_error=float(logit_cmp["max_abs_error"]),
        max_rel_error=float(logit_cmp["max_rel_error"]),
        loss_abs_error=loss_abs_error,
        logits_within_tolerance=bool(logit_cmp["within_tolerance"]),
        loss_within_tolerance=loss_within_tolerance,
    )


def run_config_parity_case(
    cfg: GPTConfigTorch,
    *,
    bridge: LeanParityBridge | None = None,
    model: MicroGPTTorch | None = None,
    checkpoint_path: str | Path | None = None,
    tokens: Sequence[int],
    predict_idx: int,
    target: int,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> ParityResult:
    bridge = bridge or LeanParityBridge()
    checkpoint_owned = False
    if checkpoint_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="config-", suffix=".bin", delete=False)
        tmp.close()
        checkpoint_path = tmp.name
        bridge.save_config_checkpoint(cfg, checkpoint_path)
        checkpoint_owned = True

    if model is None:
        model = MicroGPTTorch(cfg, dtype=torch.float64)
    load_lean_checkpoint(checkpoint_path, model)
    model.eval()

    lean_eval = bridge.eval_config(
        cfg,
        tokens=tokens,
        predict_idx=predict_idx,
        target=target,
        checkpoint=checkpoint_path,
    )
    token_tensor = torch.tensor(list(tokens), dtype=torch.long)
    with torch.no_grad():
        torch_logits_tensor = model.forward(token_tensor, predict_idx)
    torch_logits = [float(value) for value in torch_logits_tensor.detach().cpu().tolist()]
    torch_loss = cross_entropy_from_logits(torch_logits_tensor, target)

    logit_cmp = compare_logits(lean_eval.logits, torch_logits, atol=atol, rtol=rtol)
    loss_abs_error = abs(lean_eval.loss - torch_loss)
    loss_within_tolerance = loss_abs_error <= atol + rtol * max(
        abs(lean_eval.loss), abs(torch_loss), 1.0e-12
    )

    if checkpoint_owned:
        Path(checkpoint_path).unlink(missing_ok=True)

    return ParityResult(
        fixture=f"config:{cfg.d_model}/{cfg.n_heads}/{cfg.d_head}/{cfg.d_ff}/{cfg.n_layers}/{cfg.vocab}",
        checkpoint_path=str(checkpoint_path),
        lean_logits=lean_eval.logits,
        torch_logits=torch_logits,
        lean_loss=lean_eval.loss,
        torch_loss=torch_loss,
        max_abs_error=float(logit_cmp["max_abs_error"]),
        max_rel_error=float(logit_cmp["max_rel_error"]),
        loss_abs_error=loss_abs_error,
        logits_within_tolerance=bool(logit_cmp["within_tolerance"]),
        loss_within_tolerance=loss_within_tolerance,
    )
