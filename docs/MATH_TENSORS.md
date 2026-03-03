# Tensors, Shapes, and Typed Linear Algebra

## Intuition

In ordinary ML code, tensors are often just arrays with shape comments in the
developer's head. In this repo, many tensor dimensions are part of the type
itself, so shape mistakes become compile-time errors.

## Core objects

### `Vec n`

A vector of length `n`.

Think of it as:
- a one-dimensional list of numbers
- with the length carried in the type

### `Mat rows cols`

A matrix with:
- `rows` rows
- `cols` columns

### `TokenSeq len dim`

A fixed-length sequence of token vectors:
- `len` positions
- each token represented by a vector of size `dim`

## Formula view

### Dot product

For vectors `a, b in R^n`:

`a · b = sum_i a_i b_i`

Interpretation:
- measures alignment
- is the building block for attention scores and linear algebra

### Matrix-vector multiplication

For `M in R^(m x n)` and `v in R^n`:

`Mv in R^m`

This is why the type

`Mat m n -> Vec n -> Vec m`

is so useful: it says exactly which shapes can legally multiply.

### Matrix-matrix multiplication

For `A in R^(m x n)` and `B in R^(n x p)`:

`AB in R^(m x p)`

The middle dimension must match. In this repo, that compatibility is encoded in
the function types rather than left to runtime errors.

## Worked example

Take:

`M = [[1,2,3],[4,5,6]]`

and

`v = [1,1,1]`

Then:

`Mv = [1+2+3, 4+5+6] = [6,15]`

This exact example appears in the smoke suite.

## Where it appears in this repo

Main implementations:
- `Vec`
- `Mat`
- `TokenSeq`
- `Vec.dot`
- `Mat.mulVec`
- `Mat.mulMat`
- `Mat.transpose`

All live in [MicroGPT.lean](../MicroGPT.lean).

## What Lean guarantees vs what it does not

### Guaranteed

- vector lengths line up
- matrix dimensions line up
- token sequence shapes line up
- illegal residual additions do not type-check

### Not guaranteed

- that your numbers are numerically stable
- that your formulas are semantically correct
- that floating-point arithmetic behaves like ideal real arithmetic

## Why this matters for learning

If you understand these typed tensor objects, you already understand a large
part of transformer math mechanically:
- what shapes move through the network
- which operations are legal
- where information is being projected and recombined
