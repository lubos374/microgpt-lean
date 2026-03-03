# Normalization, Activations, and Loss

## Layer normalization

### Intuition

Layer norm makes one token's feature vector more numerically stable by:
- subtracting its mean
- dividing by its standard deviation
- then applying learned rescaling and shifting

### Formula

For token vector `x`:

`mean = (1/d) sum_i x_i`

`var = (1/d) sum_i (x_i - mean)^2`

`norm_i = (x_i - mean) / sqrt(var + eps)`

`output_i = gamma_i * norm_i + beta_i`

### Why it helps

Training deep models is easier when features stay in a predictable numeric
range. Layer norm helps avoid unstable internal scales.

## ReLU and GELU

### ReLU

`ReLU(x) = max(0, x)`

Simple and sparse, but with a hard kink.

### GELU

GELU is a smoother nonlinearity used in GPT-style models. It keeps the idea of
"gate small values, keep useful large values" but without the hard cutoff.

This repo uses a common approximate GELU formula.

## Logits vs probabilities

The model produces **logits** first.

Logits are unnormalized scores:
- positive means "more likely"
- negative means "less likely"
- but they are not probabilities yet

Softmax converts logits into probabilities.

## Softmax

For logits `z_i`:

`softmax(z)_i = exp(z_i) / sum_j exp(z_j)`

This repo uses a numerically stable version by subtracting the maximum logit
before exponentiating.

## Cross-entropy loss

If the correct next token is `y`, then:

`loss = -log(softmax(logits)_y)`

Interpretation:
- if the model assigns high probability to the correct token, loss is small
- if it assigns low probability, loss is large

This is why cross-entropy is the standard next-token prediction loss.

## Why define the loss on logits?

Numerically, it is better to work from logits and apply stable softmax / log
internally than to pass already-softmaxed probabilities around and risk more
rounding issues.

That is why the repo's forward contract distinguishes logits from probabilities.

## Float caveat

The formulas above are usually explained over ideal real numbers. The actual
repo uses `Float`, which means:
- rounding exists
- equalities are only approximate numerically
- proving ideal mathematical identities is harder than proving type safety

## Where it appears in this repo

See [MicroGPT.lean](../MicroGPT.lean):
- `softmax`
- `relu`
- `gelu`
- `layerNorm`
- `crossEntropyLoss`

## What Lean guarantees vs what it does not

### Guaranteed

- dimensions of normalization and loss inputs line up
- logits/loss functions are wired consistently through the typed model

### Not guaranteed today

- formal proof that softmax sums to `1`
- formal proof that cross-entropy is nonnegative
- formal proof that layer norm has ideal mean `0` and variance `1`
