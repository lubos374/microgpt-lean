# Architecture

## Overview

This repo is intentionally split into two layers:

1. **Lean** for the typed reference implementation
2. **PyTorch** for practical training and experimentation

That split is central to the project.

## Lean reference layer

The main Lean implementation lives in [MicroGPT.lean](../MicroGPT.lean).

Responsibilities:
- typed tensors and sequences
- forward pass
- autodiff implementation
- smoke tests
- fixture checkpoint export/evaluation

Why Lean is useful here:
- dimensions are explicit in types
- many shape errors become impossible to write
- the implementation acts as a disciplined executable specification

## PyTorch mirror

The mirrored backend lives under [python_backend](../python_backend).

Responsibilities:
- practical training loops
- checkpoint reading/writing
- fast evaluation
- parity against Lean

Why not train everything directly in Lean:
- the pure Lean path is too slow for practical full-corpus runs
- PyTorch gives a fast numeric backend while Lean remains the spec

## Checkpoint bridge

Lean and PyTorch share a binary checkpoint layout.

That allows:
- Lean fixture export
- PyTorch loading
- PyTorch training
- export back to Lean-compatible format

This is what makes parity testing meaningful.

## Parity flow

The normal parity cycle is:

1. load the same checkpoint in Lean and PyTorch
2. feed the same tokens
3. compare logits
4. compare loss
5. confirm tolerance

If parity breaks, the fast backend may have drifted from the spec.

## Experimental `self_edit/` path

The [self_edit](../self_edit) directory contains declaration-level edit-policy
preparation and related tooling.

This is experimental.
It is **not** part of the core verified release claim.

## What the architecture gives you

- Lean for structure and trust
- PyTorch for speed
- parity as the bridge
- a clear separation between formal guarantees and empirical validation
