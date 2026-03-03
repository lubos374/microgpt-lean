# Attention

## Intuition

Attention answers this question:

> For the current token, which earlier tokens are most relevant?

Instead of using a fixed-size window or fixed weights, attention computes
weights dynamically from the content of the sequence.

## Query, key, and value

Each token vector `x` is projected three times:

- `q = W_q x`
- `k = W_k x`
- `v = W_v x`

Interpretation:
- query: what this token is looking for
- key: what this token offers
- value: the information this token contributes if selected

## Score computation

For query token `i` and context token `j`:

`score(i, j) = (q_i · k_j) / sqrt(d_head)`

Why divide by `sqrt(d_head)`?
- without scaling, dot products grow with dimension
- large scores make softmax too sharp

## Causal masking

This repo is autoregressive, so a token must not attend to future positions.

The model applies a mask before softmax:
- allowed positions keep their score
- future positions get a large negative value

After softmax, future positions effectively get weight `0`.

## Softmax over scores

The raw scores become attention weights:

`alpha(i, j) = softmax(score(i, ·))_j`

These weights sum to `1` across the context positions for one query.

## Weighted value sum

The output for one head is:

`head_i = sum_j alpha(i, j) v_j`

So attention is a weighted average of value vectors, where the weights depend
on learned similarity.

## Multi-head attention

One head can focus on one kind of relationship. Multi-head attention runs
several heads in parallel, then concatenates their outputs and projects the
result back to model dimension.

Intuition:
- one head may focus on nearby syntax
- another may focus on repeated names
- another may focus on structural tokens

## Worked example

Suppose a query strongly matches one earlier token and weakly matches others.
Softmax will assign most of the probability mass to that one matching token, so
the output becomes close to that token's value vector.

That is why attention can be thought of as content-addressed lookup.

## Where it appears in this repo

Main pieces in [MicroGPT.lean](../MicroGPT.lean):
- `AttentionHead`
- `MultiHeadAttention`
- `attention`
- `multiHeadAttention`
- causal masking logic in the transformer path

## What Lean guarantees vs what it does not

### Guaranteed

- head dimensions line up
- multi-head concatenation matches the expected output size
- query/key/value projections are dimension-compatible

### Not guaranteed

- softmax semantic properties as formal theorems
- attention correctness as a mathematical proof
- numerical stability beyond the chosen implementation

## Why this matters for learning

Once attention is understood as:
- project
- score
- normalize
- weighted sum

the transformer stops looking mystical and becomes a structured linear-algebra
pipeline.
