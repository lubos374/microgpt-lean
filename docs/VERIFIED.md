# What Is Verified

This file is the canonical truth table for claims in this repo.

## Tier A — Type-level guarantees from Lean

These are the strongest guarantees currently present.

- `Vec n`, `Mat rows cols`, and `TokenSeq len dim` carry dimensions in types.
- A large class of shape mismatches becomes unrepresentable.
- Residual additions preserve dimensions by construction.
- Attention layout constraints such as head splitting are enforced structurally.
- The main file [MicroGPT.lean](../MicroGPT.lean) type-checks with no `sorry`
  and no `axiom`.

What this means:
- Lean rejects many malformed tensor programs at compile time.

What this does **not** mean:
- Lean has fully proved the semantic correctness of each transformer operation.

## Tier B — Formal theorem(s)

Current theorem count in the main Lean file:
- `listReplaceAt_length`

What it proves:
- replacing an element in a list preserves list length

Why it matters:
- it shows the proof style is present in the repo
- it does **not** mean the numerical transformer properties are already proved

## Tier C — Empirical validation

These checks are strong and useful, but they are not theorem proofs.

- reverse-mode autodiff agrees with finite differences on many small fixtures
- Lean↔PyTorch parity matches logits and loss within tight tolerance
- checkpoint save/load preserves exact float bits
- smoke-suite training examples reduce loss
- the Lean smoke suite exits successfully

Evidence sources:
- [MicroGPT.lean](../MicroGPT.lean) smoke suite
- [python_backend/scripts/parity_check.py](../python_backend/scripts/parity_check.py)
- [python_backend/microgpt_torch/parity.py](../python_backend/microgpt_torch/parity.py)

## Not formally proved yet

These remain future work.

- softmax properties such as normalization and nonnegativity
- cross-entropy nonnegativity
- layer norm mean/variance properties
- end-to-end AD correctness
- training convergence or guaranteed loss decrease
- self-modification invariants

## Claim-to-evidence map

| Claim | Status | Evidence |
| --- | --- | --- |
| Compile-time shape safety | Type-level guarantee | Lean types in `Vec`, `Mat`, `TokenSeq` |
| No `sorry` in main file | Direct fact | `MicroGPT.lean` |
| Lean smoke suite passes | Empirical | `lean --run MicroGPT.lean` |
| PyTorch matches Lean logits/loss | Empirical | `python_backend/scripts/parity_check.py` |
| AD is numerically consistent on tested cases | Empirical | finite-difference checks in `MicroGPT.lean` |
| Transformer semantics fully proved | Not true today | not claimed |

## Public wording to prefer

- compile-time shape safety
- type-level dimension guarantees
- empirical finite-difference validation
- Lean↔PyTorch parity

## Public wording to avoid

- fully formally verified transformer
- proved backpropagation
- proved training correctness
- verified self-improving AI
