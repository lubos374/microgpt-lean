# Abstract

MicroGPT Lean 4 is an educational implementation of a GPT-style transformer in
Lean 4 with tensor dimensions carried in dependent types. The project shows how
core transformer components such as embeddings, attention, layer
normalization, MLP blocks, logits, and cross-entropy can be written in a way
that makes many shape errors impossible to express. The Lean file contains no
`sorry` placeholders or axioms, and the implementation type-checks as a single
executable specification.

The repo does not claim full formal verification of transformer semantics or
automatic differentiation. Instead, it combines type-level guarantees with
empirical validation: reverse-mode gradients are checked against finite
differences, checkpoints round-trip exactly, and a mirrored PyTorch backend
maintains logit/loss parity with the Lean reference at tight numerical
tolerance. This split keeps Lean as the specification and verification layer
while letting PyTorch handle practical training.

The intended value of the project is educational and methodological. It gives
readers a concrete way to study transformer math, shape-safe modeling, and the
boundary between what is formally guaranteed, what is theorem-proved, and what
is only validated by tests and numerical agreement.
