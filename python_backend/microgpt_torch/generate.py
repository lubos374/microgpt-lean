from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .model import MicroGPTTorch
from .train import decode_utf8_bytes, encode_utf8_bytes


def _apply_top_k(logits: Tensor, top_k: int | None) -> Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=indices, src=values)
    return filtered


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> int:
    if logits.dim() != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    if greedy:
        return int(torch.argmax(logits).item())
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = _apply_top_k(logits / temperature, top_k)
    probs = torch.softmax(scaled, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(token.item())


@torch.no_grad()
def generate_token_ids(
    model: MicroGPTTorch,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    context: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if context <= 0:
        raise ValueError(f"context must be positive, got {context}")

    device = model.embedding.device if device is None else device
    generated = [int(token) for token in prompt]
    for _ in range(max_new_tokens):
        window = generated[-context:]
        tokens = torch.tensor(window, dtype=torch.long, device=device)
        logits = model.forward(tokens, len(window) - 1)
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
            generator=generator,
        )
        generated.append(next_token)
    return generated


@torch.no_grad()
def generate_text(
    model: MicroGPTTorch,
    prompt: str,
    *,
    max_new_tokens: int,
    context: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> tuple[list[int], str]:
    token_ids = encode_utf8_bytes(prompt)
    generated_ids = generate_token_ids(
        model,
        token_ids,
        max_new_tokens=max_new_tokens,
        context=context,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        device=device,
        generator=generator,
    )
    return generated_ids, decode_utf8_bytes(generated_ids)
