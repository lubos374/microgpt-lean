# MicroGPT Lean 4

MicroGPT Lean 4 is a tiny GPT-style transformer implemented in Lean 4 with
tensor shapes carried in dependent types. The repo combines a typed Lean
reference implementation with a mirrored PyTorch backend, so readers can study
transformer math, compile-time shape safety, empirical autodiff validation, and
Lean↔PyTorch parity in one place. It is an educational and experimental
project.

## Quickstart

```bash
elan default leanprover/lean4:v4.28.0
python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install -r python_backend/requirements.txt
make test
```

## What Is Verified

This repo separates three kinds of confidence:

- **Tier A — type-level guarantees**
  - shape-safe vectors, matrices, and token sequences
  - many dimension mismatches become compile-time errors
  - the main Lean file has no `sorry` and no axioms
- **Tier B — formal theorem(s)**
  - currently small in scope; the repo is not claiming broad theorem-level
    verification of transformer semantics
- **Tier C — empirical validation**
  - finite-difference checks
  - Lean↔PyTorch parity
  - smoke-suite loss decreases
  - checkpoint round-trip exactness

See [docs/VERIFIED.md](./docs/VERIFIED.md) for the precise claim table.

## What This Repo Teaches

This repo is useful for learning:

- transformer math through concrete code paths
- how dependent types can enforce tensor shapes
- how autodiff can be validated empirically
- how a Lean reference implementation can coexist with a PyTorch training
  backend
- where the line sits between formal guarantees and ordinary testing

## Math Guide

The math walkthroughs are written for technical readers who are curious about
LLMs but not assumed to be mathematicians or Lean experts.

- [Math overview](./docs/MATH_OVERVIEW.md)
- [Tensors and shapes](./docs/MATH_TENSORS.md)
- [Attention](./docs/MATH_ATTENTION.md)
- [Normalization and loss](./docs/MATH_NORMALIZATION_AND_LOSS.md)
- [Autodiff and training](./docs/MATH_AUTODIFF_AND_TRAINING.md)
- [Architecture](./docs/ARCHITECTURE.md)

## What Is Not Claimed

This repo does **not** claim anything just purely educational experimental project.

- a fully formally verified transformer
- a formal proof of autodiff correctness
- a formal proof of training convergence
- a verified self-modifying system
- a practical large-scale pure-Lean training engine

## Reproducibility

The public verification flow is:

- `make check` — Lean type-check
- `make smoke` — Lean smoke suite
- `make pycheck` — Python syntax check
- `make parity` — PyTorch↔Lean parity using the included checkpoint
- `make test` — all of the above

Longer command examples live in [docs/USAGE.md](./docs/USAGE.md).

## Repository Layout

- [`MicroGPT.lean`](./MicroGPT.lean) — main Lean implementation and smoke suite
- [`python_backend/`](./python_backend/) — PyTorch mirror, parity, and training backend
- [`docs/`](./docs/) — educational math and reproducibility docs
- [`self_edit/`](./self_edit/) — experimental edit-policy preparation
- [`MicroGPT-Blueprint.md`](./MicroGPT-Blueprint.md) — roadmap and research log

## Experimental Components

The [`self_edit/`](./self_edit/) directory and related edit-policy scripts are
experimental. They are not part of the core verified release claim.

## Abstract

See [ABSTRACT.md](./ABSTRACT.md).

## License

MIT. See [LICENSE](./LICENSE).
