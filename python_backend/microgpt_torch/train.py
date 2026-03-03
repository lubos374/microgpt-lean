from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .checkpoint import save_lean_checkpoint
from .config import GPTConfigTorch
from .model import MicroGPTTorch


@dataclass(frozen=True, slots=True)
class TokenExample:
    tokens: list[int]
    predict_idx: int
    target: int


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: list[TokenExample]
    valid: list[TokenExample]


@dataclass(frozen=True, slots=True)
class TextSplit:
    train: list[str]
    valid: list[str]


@dataclass(frozen=True, slots=True)
class TrainingLog:
    step: int
    train_loss: float
    valid_loss: float
    tokens_per_second: float
    train_eval_examples: int
    valid_eval_examples: int
    lr: float
    grad_norm: float


SELF_SOURCE_REPEAT_COUNT = 4

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


def repeat_strings(n: int, xs: Sequence[str]) -> list[str]:
    if n <= 0:
        return []
    return list(xs) + repeat_strings(n - 1, xs)


def load_self_source_text(repo_root: str | Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    source_path = root / "MicroGPT.lean"
    if source_path.exists():
        return source_path.read_text(encoding="utf-8")
    return "/- self source unavailable -/"


def build_priority_training_texts(self_source: str) -> list[str]:
    return (
        repeat_strings(SELF_SOURCE_REPEAT_COUNT, [self_source])
        + CURATED_MATHLIB_STYLE_SNIPPETS
        + CURATED_LEAN_PROGRAM_SNIPPETS
    )


def load_priority_training_texts(repo_root: str | Path | None = None) -> list[str]:
    return build_priority_training_texts(load_self_source_text(repo_root))


def build_small_byte_training_texts() -> list[str]:
    microgpt_self_excerpt = (
        "structure Vec (n : Nat) where\n  data : Array Float\n\n"
        "structure Mat (rows cols : Nat) where\n  data : Array (Array Float)"
    )
    return [microgpt_self_excerpt] + CURATED_MATHLIB_STYLE_SNIPPETS[:2] + CURATED_LEAN_PROGRAM_SNIPPETS[:1]


def split_dataset_every(holdout_every: int, xs: Sequence[str]) -> TextSplit:
    train: list[str] = []
    valid: list[str] = []
    for idx, item in enumerate(xs):
        if holdout_every > 0 and idx % holdout_every == 0:
            valid.append(item)
        else:
            train.append(item)
    return TextSplit(train=train, valid=valid)


def encode_utf8_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode_utf8_bytes(values: Iterable[int]) -> str:
    return bytes(values).decode("utf-8", errors="replace")


def byte_token_examples_from_stream(context: int, tokens: Sequence[int]) -> list[TokenExample]:
    if context <= 0:
        return []
    examples: list[TokenExample] = []
    for start in range(len(tokens)):
        end = start + context
        if end < len(tokens):
            window = list(tokens[start:end])
            examples.append(TokenExample(tokens=window, predict_idx=context - 1, target=int(tokens[end])))
    return examples


def byte_token_examples_from_text(context: int, text: str) -> list[TokenExample]:
    return byte_token_examples_from_stream(context, encode_utf8_bytes(text))


def byte_token_examples_from_texts(context: int, texts: Sequence[str]) -> list[TokenExample]:
    examples: list[TokenExample] = []
    for text in texts:
        examples.extend(byte_token_examples_from_text(context, text))
    return examples


def take_byte_token_examples_from_texts(context: int, limit: int, texts: Sequence[str]) -> list[TokenExample]:
    if limit <= 0:
        return []
    taken: list[TokenExample] = []
    for text in texts:
        remaining = limit - len(taken)
        if remaining <= 0:
            break
        taken.extend(byte_token_examples_from_text(context, text)[:remaining])
    return taken


def split_byte_token_examples_from_texts(
    context: int, holdout_every: int, texts: Sequence[str]
) -> DatasetSplit:
    split = split_dataset_every(holdout_every, texts)
    return DatasetSplit(
        train=byte_token_examples_from_texts(context, split.train),
        valid=byte_token_examples_from_texts(context, split.valid),
    )


def take_split_byte_token_examples_from_texts(
    context: int, holdout_every: int, n_train: int, n_valid: int, texts: Sequence[str]
) -> DatasetSplit:
    split = split_dataset_every(holdout_every, texts)
    return DatasetSplit(
        train=take_byte_token_examples_from_texts(context, n_train, split.train),
        valid=take_byte_token_examples_from_texts(context, n_valid, split.valid),
    )


def batches_of(batch_size: int, xs: Sequence[TokenExample]) -> list[list[TokenExample]]:
    if batch_size <= 0:
        return [list(xs)]
    return [list(xs[i : i + batch_size]) for i in range(0, len(xs), batch_size)]


def shuffled_batches(
    batch_size: int,
    xs: Sequence[TokenExample],
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> list[list[TokenExample]]:
    if not shuffle:
        return batches_of(batch_size, xs)
    order = torch.randperm(len(xs), generator=generator).tolist()
    shuffled = [xs[idx] for idx in order]
    return batches_of(batch_size, shuffled)


def tiny_single_head_config() -> GPTConfigTorch:
    return GPTConfigTorch(d_model=2, n_heads=1, d_head=2, d_ff=2, n_layers=1, vocab=3)


def tiny_two_layer_config() -> GPTConfigTorch:
    return GPTConfigTorch(d_model=2, n_heads=2, d_head=1, d_ff=2, n_layers=2, vocab=3)


def byte_config(
    *,
    d_model: int = 16,
    n_heads: int = 1,
    d_head: int = 16,
    d_ff: int = 32,
    n_layers: int = 1,
    vocab: int = 256,
) -> GPTConfigTorch:
    return GPTConfigTorch(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        d_ff=d_ff,
        n_layers=n_layers,
        vocab=vocab,
    )


def default_byte_example() -> TokenExample:
    texts = build_small_byte_training_texts()
    split = split_byte_token_examples_from_texts(8, 3, texts)
    if not split.train:
        raise ValueError("byte fixture example unavailable")
    return split.train[0]


def token_examples_to_tensors(
    examples: Sequence[TokenExample], *, device: torch.device | str | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    if not examples:
        raise ValueError("expected at least one training example")
    token_tensor = torch.tensor([example.tokens for example in examples], dtype=torch.long, device=device)
    predict_idx = torch.tensor([example.predict_idx for example in examples], dtype=torch.long, device=device)
    target = torch.tensor([example.target for example in examples], dtype=torch.long, device=device)
    return token_tensor, predict_idx, target


@torch.no_grad()
def evaluate_loss(
    model: MicroGPTTorch,
    dataset: Sequence[TokenExample],
    *,
    batch_size: int = 32,
    limit: int | None = None,
    device: torch.device | str | None = None,
) -> float:
    if not dataset:
        return 0.0
    total_loss = 0.0
    total_examples = 0
    eval_dataset = dataset if limit is None or limit <= 0 else dataset[:limit]
    for batch in batches_of(batch_size, eval_dataset):
        tokens, predict_idx, target = token_examples_to_tensors(batch, device=device)
        logits = model.forward_batch(tokens, predict_idx)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
        total_loss += float(loss.item())
        total_examples += len(batch)
    return total_loss / total_examples


def lr_scale_for_step(local_step: int, total_steps: int, warmup_steps: int, min_lr_scale: float) -> float:
    if total_steps <= 0:
        return 1.0
    floor = min(max(min_lr_scale, 0.0), 1.0)
    if warmup_steps > 0 and local_step <= warmup_steps:
        return max(local_step / warmup_steps, 1.0e-8)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (local_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (1.0 - floor) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
    return math.sqrt(total_sq)


def save_training_state(
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_valid_loss: float | None,
) -> None:
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_valid_loss": best_valid_loss,
        },
        Path(path),
    )


def read_training_state_metadata(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[int | None, float | None]:
    state = torch.load(Path(path), map_location=device)
    return state.get("step"), state.get("best_valid_loss")


def load_training_state(
    path: str | Path,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
) -> tuple[int | None, float | None]:
    state = torch.load(Path(path), map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    return state.get("step"), state.get("best_valid_loss")


def train_byte_model(
    model: MicroGPTTorch,
    train_examples: Sequence[TokenExample],
    valid_examples: Sequence[TokenExample],
    *,
    start_step: int = 0,
    steps: int,
    lr: float,
    batch_size: int,
    log_every: int,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | Path | None = None,
    train_eval_limit: int | None = None,
    valid_eval_limit: int | None = None,
    save_best_on_valid: bool = False,
    grad_clip_norm: float | None = None,
    warmup_steps: int = 0,
    min_lr_scale: float = 0.1,
    shuffle_train: bool = True,
    seed: int = 0,
    resume_optimizer_state_path: str | Path | None = None,
    save_optimizer_state: bool = True,
    device: torch.device | str | None = None,
) -> list[TrainingLog]:
    if not train_examples:
        raise ValueError("train_examples must be non-empty")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logs: list[TrainingLog] = []
    batch_generator = torch.Generator()
    batch_generator.manual_seed(seed)
    train_batches = shuffled_batches(batch_size, train_examples, shuffle=shuffle_train, generator=batch_generator)

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_valid_loss: float | None = None
    if resume_optimizer_state_path is not None:
        _, restored_best = load_training_state(
            resume_optimizer_state_path,
            optimizer,
            device=device,
        )
        if restored_best is not None:
            best_valid_loss = restored_best
    log_path = checkpoint_root / "train_log.jsonl" if checkpoint_root is not None else None

    for local_step in range(1, steps + 1):
        global_step = start_step + local_step
        batch_index = (local_step - 1) % len(train_batches)
        if batch_index == 0 and local_step > 1:
            train_batches = shuffled_batches(batch_size, train_examples, shuffle=shuffle_train, generator=batch_generator)
        batch = train_batches[batch_index]
        tokens, predict_idx, target = token_examples_to_tensors(batch, device=device)
        step_start = perf_counter()
        current_lr = lr * lr_scale_for_step(local_step, steps, warmup_steps, min_lr_scale)
        set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_batch(tokens, predict_idx)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm).item())
        else:
            grad_norm = total_grad_norm(model.parameters())
        optimizer.step()
        step_elapsed = max(perf_counter() - step_start, 1.0e-9)

        if log_every > 0 and global_step % log_every == 0:
            train_loss = evaluate_loss(
                model,
                train_examples,
                batch_size=batch_size,
                limit=train_eval_limit,
                device=device,
            )
            valid_loss = evaluate_loss(
                model,
                valid_examples,
                batch_size=batch_size,
                limit=valid_eval_limit,
                device=device,
            )
            tokens_per_second = (len(batch) * len(batch[0].tokens)) / step_elapsed
            logs.append(
                TrainingLog(
                    step=global_step,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    tokens_per_second=tokens_per_second,
                    train_eval_examples=len(train_examples) if train_eval_limit is None or train_eval_limit <= 0 else min(len(train_examples), train_eval_limit),
                    valid_eval_examples=len(valid_examples) if valid_eval_limit is None or valid_eval_limit <= 0 else min(len(valid_examples), valid_eval_limit),
                    lr=current_lr,
                    grad_norm=grad_norm,
                )
            )
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "train_loss": train_loss,
                                "valid_loss": valid_loss,
                                "tokens_per_second": tokens_per_second,
                                "train_eval_examples": len(train_examples) if train_eval_limit is None or train_eval_limit <= 0 else min(len(train_examples), train_eval_limit),
                                "valid_eval_examples": len(valid_examples) if valid_eval_limit is None or valid_eval_limit <= 0 else min(len(valid_examples), valid_eval_limit),
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                            }
                        )
                        + "\n"
                    )
            if checkpoint_root is not None and save_best_on_valid:
                if best_valid_loss is None or valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    save_lean_checkpoint(checkpoint_root / "best.bin", model)
                    (checkpoint_root / "best.txt").write_text(
                        f"step={global_step}\nvalid_loss={valid_loss:.9f}\nlr={current_lr:.12f}\ngrad_norm={grad_norm:.9f}\n",
                        encoding="utf-8",
                    )

        if checkpoint_root is not None and checkpoint_every is not None and checkpoint_every > 0:
            if global_step % checkpoint_every == 0:
                step_checkpoint = checkpoint_root / f"step-{global_step:06d}.bin"
                save_lean_checkpoint(step_checkpoint, model)
                save_lean_checkpoint(checkpoint_root / "latest.bin", model)
                if save_optimizer_state:
                    save_training_state(
                        step_checkpoint.with_suffix(".optim.pt"),
                        optimizer=optimizer,
                        step=global_step,
                        best_valid_loss=best_valid_loss,
                    )
                    save_training_state(
                        checkpoint_root / "latest.optim.pt",
                        optimizer=optimizer,
                        step=global_step,
                        best_valid_loss=best_valid_loss,
                    )

    if checkpoint_root is not None:
        save_lean_checkpoint(checkpoint_root / "latest.bin", model)
        if save_optimizer_state:
            save_training_state(
                checkpoint_root / "latest.optim.pt",
                optimizer=optimizer,
                step=start_step + steps,
                best_valid_loss=best_valid_loss,
            )

    return logs
