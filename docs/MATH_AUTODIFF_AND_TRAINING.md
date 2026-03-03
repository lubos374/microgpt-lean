# Autodiff and Training

## Why gradients matter

Training means adjusting parameters so the loss goes down.

To do that, we need derivatives:

`d(loss) / d(parameter)`

Those derivatives tell us how sensitive the loss is to each parameter.

## Chain rule

Neural networks are compositions of many functions. The derivative of the whole
pipeline comes from the chain rule:

if `f(x) = g(h(x))`, then

`f'(x) = g'(h(x)) * h'(x)`

Backpropagation is just the chain rule applied systematically to a computation
graph.

## Forward-mode AD

Forward-mode tracks:
- the value
- and its derivative with respect to one input

Good for:
- small numbers of parameters
- debugging and checking formulas

This repo uses scalar dual numbers as a stepping stone and sanity tool.

## Reverse-mode AD

Reverse-mode computes:
- all the forward values first
- then propagates gradients backward from the final loss

Why it is used for training:
- one loss
- many parameters

That is exactly the setting where reverse-mode is efficient.

## Tape-based backprop in this repo

The repo records operations in a tape-like structure and then walks it in
reverse, accumulating gradients.

Intuition:
- forward pass: remember what happened
- backward pass: assign blame for the final loss

## Finite differences

Finite differences numerically approximate derivatives:

`f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 eps)`

Why this matters:
- it does not prove your gradient is mathematically correct
- but it is a strong debugging check

This repo uses finite differences extensively as empirical validation.

## SGD and Adam

### SGD

The simplest optimizer:

`param := param - lr * grad`

### Adam

Adam keeps moving averages of:
- the gradient
- the squared gradient

It usually converges more smoothly than plain SGD.

## Why parity with PyTorch matters

The Lean implementation is the typed reference.
The PyTorch implementation is the practical training backend.

Parity means:
- same weights
- same inputs
- same logits/loss within tolerance

That gives confidence that the fast backend is still following the Lean spec.

## Where it appears in this repo

Lean side:
- [MicroGPT.lean](../MicroGPT.lean)

PyTorch side:
- [python_backend/microgpt_torch/train.py](../python_backend/microgpt_torch/train.py)
- [python_backend/microgpt_torch/parity.py](../python_backend/microgpt_torch/parity.py)

## What Lean guarantees vs what it does not

### Guaranteed today

- typed structure of many operations
- the implementation compiles and runs

### Empirically checked

- gradients on tested fixtures agree with finite differences
- PyTorch and Lean agree numerically on parity cases

### Not formally proved yet

- full AD correctness theorem
- global optimizer correctness
- convergence guarantees
