from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .config import GPTConfigTorch


def safe_nat_denom(value: int) -> float:
    return 1.0 if value == 0 else float(value)


def gelu_lean(x: Tensor) -> Tensor:
    cubic = x * x * x
    cdf = 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * cubic)))
    return x * cdf


class LayerNormTorch(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5, *, dtype: torch.dtype, device: torch.device | str | None) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta


class AttentionHeadTorch(nn.Module):
    def __init__(self, d_model: int, d_head: int, *, dtype: torch.dtype, device: torch.device | str | None) -> None:
        super().__init__()
        self.wq = nn.Parameter(torch.empty((d_head, d_model), dtype=dtype, device=device))
        self.wk = nn.Parameter(torch.empty((d_head, d_model), dtype=dtype, device=device))
        self.wv = nn.Parameter(torch.empty((d_head, d_model), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.wq, mean=0.0, std=0.02)
        nn.init.normal_(self.wk, mean=0.0, std=0.02)
        nn.init.normal_(self.wv, mean=0.0, std=0.02)


class MultiHeadAttentionTorch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.heads = nn.ModuleList(
            AttentionHeadTorch(d_model, d_head, dtype=dtype, device=device) for _ in range(n_heads)
        )
        self.wo = nn.Parameter(torch.empty((d_model, n_heads * d_head), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.wo, mean=0.0, std=0.02)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        single_sequence = x.dim() == 2
        if single_sequence:
            x = x.unsqueeze(0)

        batch, seq_len, _ = x.shape
        scale_raw = math.sqrt(float(self.d_head))
        scale = 1.0 if scale_raw == 0.0 else scale_raw
        mask = None
        if causal:
            mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
                diagonal=1,
            )

        head_outputs: list[Tensor] = []
        for head in self.heads:
            q = x @ head.wq.transpose(0, 1)
            k = x @ head.wk.transpose(0, 1)
            v = x @ head.wv.transpose(0, 1)
            scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            if mask is not None:
                scores = scores.masked_fill(mask, -1.0e9)
            weights = torch.softmax(scores, dim=-1)
            head_outputs.append(torch.matmul(weights, v))

        concatenated = torch.cat(head_outputs, dim=-1)
        output = concatenated @ self.wo.transpose(0, 1)
        if single_sequence:
            return output.squeeze(0)
        return output


class TransformerLayerTorch(nn.Module):
    def __init__(self, cfg: GPTConfigTorch, *, dtype: torch.dtype, device: torch.device | str | None) -> None:
        super().__init__()
        self.ln1 = LayerNormTorch(cfg.d_model, dtype=dtype, device=device)
        self.attn = MultiHeadAttentionTorch(
            cfg.d_model,
            cfg.n_heads,
            cfg.d_head,
            dtype=dtype,
            device=device,
        )
        self.ln2 = LayerNormTorch(cfg.d_model, dtype=dtype, device=device)
        self.mlp_fc1 = nn.Parameter(torch.empty((cfg.d_ff, cfg.d_model), dtype=dtype, device=device))
        self.mlp_fc2 = nn.Parameter(torch.empty((cfg.d_model, cfg.d_ff), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mlp_fc1, mean=0.0, std=0.02)
        nn.init.normal_(self.mlp_fc2, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        normed_seq = self.ln1(x)
        attn_out = self.attn(normed_seq, causal=True)
        x_resid = x + attn_out
        normed2 = self.ln2(x_resid)
        hidden = gelu_lean(normed2 @ self.mlp_fc1.transpose(0, 1))
        mlp_out = hidden @ self.mlp_fc2.transpose(0, 1)
        return x_resid + mlp_out


class MicroGPTTorch(nn.Module):
    def __init__(
        self,
        cfg: GPTConfigTorch,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Parameter(torch.empty((cfg.d_model, cfg.vocab), dtype=dtype, device=device))
        self.layers = nn.ModuleList(
            TransformerLayerTorch(cfg, dtype=dtype, device=device) for _ in range(cfg.n_layers)
        )
        self.ln_final = LayerNormTorch(cfg.d_model, dtype=dtype, device=device)
        self.unembed = nn.Parameter(torch.empty((cfg.vocab, cfg.d_model), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.unembed, mean=0.0, std=0.02)

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.dtype

    def positional_encoding(self, seq_len: int, *, device: torch.device | str | None = None) -> Tensor:
        if device is None:
            device = self.embedding.device
        dim = self.cfg.d_model
        idx = torch.arange(dim, dtype=torch.int64, device=device)
        pair_idx = torch.div(idx, 2, rounding_mode="floor")
        exponent = (2 * pair_idx.to(dtype=self.dtype)) / safe_nat_denom(dim)
        positions = torch.arange(seq_len, dtype=self.dtype, device=device).unsqueeze(1)
        angle = positions / torch.pow(torch.tensor(10000.0, dtype=self.dtype, device=device), exponent.unsqueeze(0))
        even_mask = (idx % 2 == 0).unsqueeze(0)
        return torch.where(even_mask, torch.sin(angle), torch.cos(angle))

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 1:
            raise ValueError(f"expected 1-D tokens tensor, got shape {tuple(tokens.shape)}")
        gathered = self.embedding.index_select(dim=1, index=tokens.to(device=self.embedding.device, dtype=torch.long))
        return gathered.transpose(0, 1)

    def embed_tokens_batch(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"expected 2-D batch tensor, got shape {tuple(tokens.shape)}")
        flat = tokens.to(device=self.embedding.device, dtype=torch.long).reshape(-1)
        gathered = self.embedding.index_select(dim=1, index=flat)
        return gathered.transpose(0, 1).reshape(tokens.shape[0], tokens.shape[1], self.cfg.d_model)

    def input_sequence(self, tokens: Tensor) -> Tensor:
        embedded = self.embed_tokens(tokens)
        return embedded + self.positional_encoding(tokens.shape[0], device=embedded.device)

    def input_sequence_batch(self, tokens: Tensor) -> Tensor:
        embedded = self.embed_tokens_batch(tokens)
        return embedded + self.positional_encoding(tokens.shape[1], device=embedded.device).unsqueeze(0)

    def hidden_states(self, tokens: Tensor) -> Tensor:
        x = self.input_sequence(tokens)
        for layer in self.layers:
            x = layer(x)
        return x

    def hidden_states_batch(self, tokens: Tensor) -> Tensor:
        x = self.input_sequence_batch(tokens)
        for layer in self.layers:
            x = layer(x)
        return x

    def output_head(self, hidden: Tensor) -> Tensor:
        normed = self.ln_final(hidden)
        return normed @ self.unembed.transpose(0, 1)

    def forward(self, tokens: Tensor, predict_idx: int) -> Tensor:
        if tokens.dim() != 1:
            raise ValueError(f"expected 1-D tokens tensor, got shape {tuple(tokens.shape)}")
        seq_len = tokens.shape[0]
        if predict_idx < 0 or predict_idx >= seq_len:
            raise IndexError(f"predict_idx {predict_idx} out of range for sequence length {seq_len}")
        hidden = self.hidden_states(tokens)[predict_idx]
        return self.output_head(hidden)

    def forward_batch(self, tokens: Tensor, predict_idx: Tensor | int) -> Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"expected 2-D batch tensor, got shape {tuple(tokens.shape)}")
        hidden = self.hidden_states_batch(tokens)
        batch = hidden.shape[0]
        if isinstance(predict_idx, int):
            if predict_idx < 0 or predict_idx >= hidden.shape[1]:
                raise IndexError(
                    f"predict_idx {predict_idx} out of range for sequence length {hidden.shape[1]}"
                )
            selected = hidden[:, predict_idx, :]
        else:
            idx = predict_idx.to(device=hidden.device, dtype=torch.long)
            if idx.dim() != 1 or idx.shape[0] != batch:
                raise ValueError(
                    f"predict_idx tensor must have shape ({batch},), got {tuple(idx.shape)}"
                )
            selected = hidden[torch.arange(batch, device=hidden.device), idx]
        return self.output_head(selected)

    def to_flat_params(self) -> list[float]:
        flat: list[float] = []
        flat.extend(self.embedding.detach().cpu().reshape(-1).tolist())
        for layer in self.layers:
            flat.extend(layer.ln1.gamma.detach().cpu().reshape(-1).tolist())
            flat.extend(layer.ln1.beta.detach().cpu().reshape(-1).tolist())
            for head in layer.attn.heads:
                flat.extend(head.wq.detach().cpu().reshape(-1).tolist())
                flat.extend(head.wk.detach().cpu().reshape(-1).tolist())
                flat.extend(head.wv.detach().cpu().reshape(-1).tolist())
            flat.extend(layer.attn.wo.detach().cpu().reshape(-1).tolist())
            flat.extend(layer.ln2.gamma.detach().cpu().reshape(-1).tolist())
            flat.extend(layer.ln2.beta.detach().cpu().reshape(-1).tolist())
            flat.extend(layer.mlp_fc1.detach().cpu().reshape(-1).tolist())
            flat.extend(layer.mlp_fc2.detach().cpu().reshape(-1).tolist())
        flat.extend(self.ln_final.gamma.detach().cpu().reshape(-1).tolist())
        flat.extend(self.ln_final.beta.detach().cpu().reshape(-1).tolist())
        flat.extend(self.unembed.detach().cpu().reshape(-1).tolist())
        return flat

    def load_flat_params(self, values: Sequence[float]) -> None:
        flat = torch.tensor(list(values), dtype=self.dtype, device=self.embedding.device)
        if flat.numel() != self.cfg.param_count:
            raise ValueError(
                f"flat parameter count mismatch: got {flat.numel()}, expected {self.cfg.param_count}"
            )

        offset = 0

        def take(count: int) -> Tensor:
            nonlocal offset
            chunk = flat[offset : offset + count]
            offset += count
            return chunk

        with torch.no_grad():
            self.embedding.copy_(take(self.cfg.embedding_param_count).view_as(self.embedding))
            for layer in self.layers:
                layer.ln1.gamma.copy_(take(self.cfg.d_model).view_as(layer.ln1.gamma))
                layer.ln1.beta.copy_(take(self.cfg.d_model).view_as(layer.ln1.beta))
                for head in layer.attn.heads:
                    head.wq.copy_(take(self.cfg.d_head * self.cfg.d_model).view_as(head.wq))
                    head.wk.copy_(take(self.cfg.d_head * self.cfg.d_model).view_as(head.wk))
                    head.wv.copy_(take(self.cfg.d_head * self.cfg.d_model).view_as(head.wv))
                layer.attn.wo.copy_(
                    take(self.cfg.d_model * (self.cfg.n_heads * self.cfg.d_head)).view_as(layer.attn.wo)
                )
                layer.ln2.gamma.copy_(take(self.cfg.d_model).view_as(layer.ln2.gamma))
                layer.ln2.beta.copy_(take(self.cfg.d_model).view_as(layer.ln2.beta))
                layer.mlp_fc1.copy_(take(self.cfg.d_ff * self.cfg.d_model).view_as(layer.mlp_fc1))
                layer.mlp_fc2.copy_(take(self.cfg.d_model * self.cfg.d_ff).view_as(layer.mlp_fc2))
            self.ln_final.gamma.copy_(take(self.cfg.d_model).view_as(self.ln_final.gamma))
            self.ln_final.beta.copy_(take(self.cfg.d_model).view_as(self.ln_final.beta))
            self.unembed.copy_(take(self.cfg.vocab * self.cfg.d_model).view_as(self.unembed))

        if offset != flat.numel():
            raise RuntimeError(f"flat parameter load ended at {offset}, expected {flat.numel()}")
