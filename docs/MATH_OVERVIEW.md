# Math Overview

This file gives a plain engineering walkthrough of the full model.

## 1. Tokens and sequences

An LLM starts with discrete token ids. In this repo, the practical training
path uses byte-level UTF-8 tokens, so each input token is an integer between
`0` and `255`.

A sequence is just an ordered list of tokens. Order matters, so the model must
carry both token identity and token position.

## 2. Embedding lookup

Each token id is mapped to a learned vector of size `d_model`.

Formula:

`embedding(token) = E[:, token]`

where `E` is a `d_model x vocab` matrix.

Intuition:
- tokens start as integers
- embeddings turn them into continuous vectors the model can transform

## 3. Positional encoding

Embeddings alone do not tell the model whether a token appeared first or last.
Positional encoding injects order information.

This repo uses sinusoidal positional encodings. Each position gets a vector of
the same size as the embedding, alternating sine and cosine terms at different
frequencies.

## 4. Attention

Attention lets each token decide which earlier tokens matter most.

Each token is projected into:
- a query `Q`
- a key `K`
- a value `V`

The attention score is a similarity between the query of the current token and
the keys of the context tokens.

Formula:

`score(i, j) = (Q_i · K_j) / sqrt(d_head)`

Then the model applies a softmax over scores to get weights and forms a
weighted sum of the value vectors.

## 5. Residual connection + layer norm

After attention, the model adds the attention output back to the input. This is
a residual connection.

Then it uses layer normalization to center and rescale the features so
optimization stays stable.

## 6. MLP block

Each transformer block also contains a feedforward network:

`x -> Linear -> GELU -> Linear`

Intuition:
- attention mixes information across tokens
- the MLP transforms information within each token representation

## 7. Logits and cross-entropy

The final hidden vector is projected back into vocabulary space, giving one
score per token:

`logits = W_unembed * h`

These scores are logits, not probabilities. Softmax can turn them into a
distribution, and cross-entropy measures how wrong the model is about the
correct next token.

## 8. Gradients and training

Training means changing parameters to reduce loss.

The repo implements automatic differentiation so it can compute gradients of
loss with respect to parameters. Those gradients are then used by optimizers
such as SGD or Adam.

## 9. Where Lean helps and where it does not

Lean helps most with structural correctness:
- dimensions
- shape compatibility
- explicit program structure

Lean does **not** automatically prove the floating-point math is semantically
correct. That is why this repo separates:
- type-level guarantees
- theorem-level proofs
- empirical checks

## Running example

For a small toy sequence:

1. token ids enter the embedding matrix
2. positional encoding is added
3. each transformer layer applies attention, residuals, normalization, and MLP
4. the final hidden state becomes logits
5. cross-entropy compares those logits to the true next token

See the more detailed math docs for each stage:
- [MATH_TENSORS.md](./MATH_TENSORS.md)
- [MATH_ATTENTION.md](./MATH_ATTENTION.md)
- [MATH_NORMALIZATION_AND_LOSS.md](./MATH_NORMALIZATION_AND_LOSS.md)
- [MATH_AUTODIFF_AND_TRAINING.md](./MATH_AUTODIFF_AND_TRAINING.md)
