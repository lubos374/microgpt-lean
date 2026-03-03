# MicroGPT Lean 4 — Full Implementation Blueprint

This blueprint is a roadmap and research log, not a statement of completed
formal verification.

## Vision

A self-modifying, formally verified transformer written entirely in Lean 4,
trained on Lean/Mathlib data, where every self-modification must pass
the Lean type checker before being accepted.

---

## What Exists (Phase 0 — Done)

```
✅ Dependently-typed Vec n, Mat rows cols
✅ matmul, dot product, transpose
✅ softmax (numerically stable), relu, gelu
✅ Layer normalization with learnable params
✅ Single-head scaled dot-product attention
✅ Multi-head attention
✅ Causal masking in transformer attention
✅ Sinusoidal positional encoding
✅ Full transformer layer (pre-norm, residuals, MLP)
✅ Model forward pass: tokens → logits
✅ Cross-entropy loss on logits
✅ Byte-level UTF-8 tokenizer helpers
✅ Greedy token generation (argmax)
✅ Temperature and top-k sampling
✅ Forward-mode DualFloat autodiff for scalar primitives
✅ Finite-difference gradient checks for scalar functions
✅ Smoke tests for tensor ops, single/multi-head attention, positional encodings, tokenizer, full model forward, and sampling
```

---

## Phase 1: Complete Forward Pass

**Goal:** A fully functional inference engine that can process sequences.

```
├── 1.1 Multi-Head Attention
│   ├── Status: implemented with typed head concatenation
│   ├── Split d_model into n_heads × d_head
│   ├── Run n_heads attention heads in parallel
│   ├── Concatenate outputs
│   └── Type: MultiHeadAttn d_model n_heads d_head
│       where d_model = n_heads * d_head (enforced by type)
│
├── 1.2 Positional Encoding
│   ├── Status: sinusoidal encoding implemented
│   ├── Sinusoidal (fixed, no params)
│   ├── OR Learned position embeddings (trainable)
│   ├── Add to token embeddings before first layer
│   └── Type: Vec d_model for each position
│
├── 1.3 Causal Masking
│   ├── Status: implemented for transformer multi-head attention
│   ├── Mask future tokens in attention scores
│   ├── Set masked positions to -∞ before softmax
│   └── Essential for autoregressive generation
│
├── 1.4 Token Generation Loop
│   ├── Status: argmax, temperature, and top-k generation implemented
│   ├── Forward pass → logits → sample next token
│   ├── Sampling strategies: argmax, temperature, top-k
│   ├── Append token, repeat
│   └── KV-cache for efficient generation (optional for MVP)
│
└── 1.5 Tokenizer
    ├── Status: UTF-8 byte encode/decode helpers implemented
    ├── Simple: byte-level (one UTF-8 byte = one token)
    ├── Vocab size = 256 for byte-level
    ├── Encode: String → List (Fin vocab) via UTF-8 bytes
    ├── Decode: List (Fin vocab) → String via UTF-8 bytes
    └── BPE is out of scope for MVP
```

**Estimated effort:** 1-2 weeks
**Dependency:** None, builds on Phase 0

---

## Phase 2: Automatic Differentiation

**Goal:** Compute gradients of any computation graph built from Lean operations.

This is the hardest engineering phase. No existing Lean autodiff library exists.

```
├── 2.1 Dual Numbers (Forward-Mode AD)
│   ├── Status: scalar DualFloat primitives implemented
│   ├── Define DualFloat: { val : Float, deriv : Float }
│   ├── Lift all operations: add, mul, exp, tanh, etc.
│   ├── Forward-mode computes one partial derivative per pass
│   ├── Simple to implement, inefficient for many parameters
│   └── Use as: stepping stone and test oracle for reverse-mode
│
├── 2.2 Tape-Based Reverse-Mode AD
│   ├── Status: scalar tape, flat-list gradients, and logits loss backprop implemented
│   ├── Record computation graph as a tape (list of operations)
│   ├── Forward: evaluate and record ops
│   ├── Backward: walk tape in reverse, accumulate gradients
│   ├── Define Traced type that wraps Float + tape entry
│   │
│   ├── Core operations to differentiate:
│   │   ├── add, mul, sub, div
│   │   ├── exp, log, sqrt, tanh
│   │   ├── matmul (→ two gradient paths)
│   │   ├── softmax (Jacobian-vector product)
│   │   ├── layer norm (3 gradient paths: input, gamma, beta)
│   │   └── cross-entropy loss
│   │
│   └── Type structure:
│       ├── Tape : Array TapeEntry
│       ├── TapeEntry : { op, inputs, output, grad_fn }
│       └── backward : Tape → GradMap
│
├── 2.3 Gradient Verification
│   ├── Status: finite-difference checks implemented for scalar ops, list matvec, and logits loss
│   ├── Numerical gradient checking (finite differences)
│   ├── Compare autodiff gradients vs numerical for each op
│   ├── This is where Lean shines: you could prove
│   │   that your AD rules are mathematically correct
│   └── Even without full proofs, property-based testing
│
└── 2.4 Gradient Computation for Full Model
    ├── Status: typed linear-head gradients, typed traced Vec/Mat helpers, frozen-feature/model-output-head/MLP/attention/full-block training, generic multi-head full-layer gradients, generic actual-model last-layer training, and generic flat full-model gradients implemented
    ├── loss(model, input, target) → Float
    ├── grad(loss, model.params) → typed gradients
    ├── Current scope: linear `Wx + b` head with row-major parameter packing
    ├── Bridge steps:
    │   ├── train a fresh output head on top of frozen MicroGPT hidden states
    │   └── train the real `ln_final + unembed` output head on frozen transformer states
    ├── MLP block backprop now works against a fixed model output head
    ├── Single-head attention block backprop now works against a fixed output head
    ├── One full frozen-sequence transformer block now backprops end to end against a fixed output head
    ├── Single-layer single-head MicroGPT block training is now wired into an actual model instance
    ├── Generic multi-head full-layer gradients now match finite differences
    ├── Generic actual-model last-layer training now works for non-empty multi-layer `MicroGPT` models
    ├── Generic full-model flat packing/unpacking now covers embeddings, all layers, final norm, and unembedding
    ├── Generic end-to-end full-model reverse gradients now match finite differences on a tiny actual model
    ├── Remaining gap: richer optimizers/checkpointing and scaling the same path to broader byte-token training
    └── Test: typed gradients match numerical checks and one-example SGD lowers loss
```

**Estimated effort:** 3-6 weeks
**Dependency:** Phase 1 (need forward pass to differentiate)
**This is the make-or-break phase.** If autodiff works, everything after is incremental.

---

## Phase 3: Training Loop

**Goal:** Update weights via gradient descent. Train on actual data.

```
├── 3.1 Loss Function
│   ├── Cross-entropy loss over vocabulary
│   ├── Input: predicted logits (Vec vocab) + target token (Fin vocab)
│   ├── loss = -log(softmax(logits)[target])
│   └── Type-safe: target must be valid vocab index
│
├── 3.2 Optimizer
│   ├── Status: SGD and Adam implemented over flat parameter vectors, including generic full-model Adam updates, learning-rate warmup/cosine decay, gradient clipping, optimizer-state resume sidecars, and epoch-wise/logged actual-model runs
│   ├── Status: the practical PyTorch trainer now resumes from `latest.bin` plus `.optim.pt` sidecars without losing global step metadata
│   ├── SGD (simplest, implement first)
│   │   └── param -= lr * grad
│   ├── Adam
│   │   ├── First moment (mean of gradients)
│   │   ├── Second moment (mean of squared gradients)
│   │   ├── Bias correction
│   │   └── 4 hyperparams: lr, beta1, beta2, eps
│   └── All params stored as flat Vec for simplicity
│
├── 3.3 Data Pipeline
│   ├── Status: priority corpus strategy documented; byte-token next-token example builder, deterministic train/validation split and batching helpers, and a real `vocab = 256` byte-model smoke fixture implemented
│   ├── Priority 1: MicroGPT's own source code, oversampled so the model memorizes its own structure
│   ├── Priority 2: short contextual Lean proofs/defs (not isolated tactic strings)
│   ├── Priority 3: small Lean 4 programs, preferring files under ~500 lines
│   ├── Exclude: natural-language math, LaTeX, and non-Lean languages
│   ├── Byte-level tokenization over UTF-8
│   ├── Chunk into fixed-length next-token examples
│   ├── Deduplicate aggressively and split validation by file, not random chunks
│   └── Simple start: self source + curated snippets → split texts by file → byte-token examples → batches
│
├── 3.4 Training Loop
│   ├── Status: single-example recursive SGD loops implemented for the tiny linear classifier, frozen-feature output head, frozen-state model output head, frozen-state MLP block, frozen-sequence attention block, frozen-sequence full transformer block, a tiny actual-model token loop, a generic last-layer batched/logged actual-model loop, a generic full-model batched/logged actual-model loop, a generic full-model Adam loop with epoch logging, reshuffled minibatches, machine-readable JSONL training logs, a byte-token `vocab = 256` actual-model smoke run, and native byte-model Adam experiment entry points
│   ├── for epoch in epochs:
│   │   ├── for batch in data:
│   │   │   ├── forward pass → logits
│   │   │   ├── compute loss
│   │   │   ├── backward pass → gradients
│   │   │   ├── optimizer step
│   │   │   └── log loss
│   │   └── evaluate on held-out data
│   └── Save checkpoints (serialize weights to file) via flat binary parameter checkpoints
│
├── 3.5 Weight Serialization
│   ├── Status: binary checkpoint save/load implemented for flat full-model parameters, with PyTorch-only optimizer-state sidecars for practical resume
│   ├── Save: model params → binary file `[paramCount:u64][float_bits...]`
│   ├── Load: file → model params, validated against target config
│   └── Enables checkpointing and resuming training
│
├── 3.6 Training Verification
│   ├── Overfit on a single tiny sequence (must reach ~0 loss)
│   ├── Loss should decrease monotonically on tiny data
│   ├── Status: tiny actual-model Adam smoke test verified loss decrease from `6.050875` to `4.582892` over 10 steps
│   ├── Status: native byte-model Adam quick probe lowered loss from `5.815905` to `5.379868` after 1 step; 2-step timing probe took `11.42s` on CPU for a full-model byte run over a streamed subset from a `539347`-example training split
│   ├── Generated output should memorize training data at small scale
│   └── If these fail, debug autodiff first
│
├── 3.7 Practical Compute Backend
│   ├── Status: initial PyTorch backend implemented under `python_backend/`
│   ├── Status: Lean parity CLI implemented in `MicroGPT.lean` via `parity-save`, `parity-eval`, and `parity-default-case`
│   ├── Status: checkpoint reader/writer, flat parameter mirroring, parity harness, and byte-training script added on the Python side
│   ├── Status: checkpoint sampling script and capped-eval long-run training options added on the Python side
│   ├── Status: resumed training and batch checkpoint evaluation tooling added on the Python side
│   ├── Status: exact next-byte accuracy evaluation added, plus `config.json` sidecars for non-default PyTorch model sizes
│   ├── Status: the first larger PyTorch-only byte model (`d_model=32, n_heads=2, d_head=16, d_ff=64`) outperformed the `d_model=16` baseline on held-out next-byte accuracy
│   ├── Status: Lean parity now supports arbitrary byte-model configs via `parity-save-config` and `parity-eval-config`
│   ├── Status: exact best-checkpoint selection by held-out accuracy and fixed benchmark prompts added on the Python side
│   ├── Status: the first stabilized 2-layer byte model (`d_model=32, n_heads=2, d_head=16, d_ff=64, n_layers=2, context=16`) trained cleanly with warmup/clipping, resumed correctly from optimizer sidecars, preserved Lean parity, and at `20k` steps beat the 1-layer `d_model=32` baseline with held-out accuracy `0.538330` vs `0.434082`
│   ├── Status: a follow-up fine-tune of that 2-layer model on `context=32` produced a parity-safe `context=32` checkpoint with held-out accuracy `0.531982`; this strongly improved same-context performance over the unfine-tuned `context=16` checkpoint on `context=32` validation (`0.122559`), but did not beat the `context=16` run on its own home metric
│   ├── Keep Lean as the executable specification and checkpoint verifier
│   ├── Port the exact architecture and flat parameter layout to PyTorch
│   ├── Add parity tests: same weights + same tokens → matching logits/loss in Lean and PyTorch
│   ├── Train in PyTorch/Colab for speed; export weights back into Lean's flat checkpoint format
│   ├── Use Lean for inference checks, typed guarantees, and future proof obligations
│   └── Rationale: the current pure-Lean full-model byte training path is correct enough to specify behavior, but too slow for practical full-corpus training
│
└── 3.8 Edit-Policy Preparation
    ├── Status: initial declaration-level edit dataset tooling implemented under `self_edit/` plus `python_backend/scripts/build_edit_dataset.py`
    ├── Status: prompt/response edit-policy evaluation and one-declaration proposal scripts added via `python_backend/scripts/evaluate_edit_policy.py` and `python_backend/scripts/propose_edit.py`
    ├── Status: edit-policy fine-tuning script added via `python_backend/scripts/train_edit_policy.py`
    ├── Status: prompt-locality bugs were identified and partially fixed; the prompt now ends with a declaration-specific replacement marker instead of a generic suffix
    ├── Status: edit-policy held-out response-token accuracy improved from `0.584355` on the initial setup to `0.642676` on the current `v3` prompt/fine-tune path
    ├── Status: a `context=32` follow-up recovered from a catastrophic zero-shot transfer baseline (`accuracy=0.108264`) to a trained `context=32` checkpoint at `accuracy=0.563021`, but this still underperformed the `context=16` `v3` best and therefore did not replace it
    ├── Status: proposal generation now supports multi-sample stochastic decoding plus basic structural filters, but sampled bodies are still mostly repo-shaped fragments rather than valid Lean replacements
    ├── Goal: turn the current byte model from a raw continuation model into an edit proposer
    ├── Build declaration-level training and evaluation data
    ├── Define edit prompts and a strict response format
    ├── Add checkpoint selection for edit usefulness, not only next-byte accuracy
    ├── Keep proposal generation grounded in actual repository declarations
    └── Dependency for Phase 4: self-modification should consume an edit policy, not a generic byte predictor
```

**Estimated effort:** 2-3 weeks
**Dependency:** Phase 2 (need gradients)

---

## Phase 4: Self-Modification Engine

**Goal:** Build a verifier-gated edit search loop over a cloned copy of the repo.
The model proposes local code edits, MCTS chooses which edits to evaluate,
and Lean plus task-specific checks decide reward and acceptance.

This is the research frontier. The v1 system is intentionally conservative.

### 4.1 Objective

- Repository target: this repo only
- Proposal policy: current byte-level model, upgraded into an edit proposer via Phase 3.8
- Search strategy: MCTS over local edits
- Acceptance priority: compile-first
- Training target: improve measured code/task fitness, not language-model loss directly

### 4.2 Initial Scope

- Editable surface: one function/theorem/def body per action
- Search depth: `1` edit per rollout
- Supported declaration kinds:
  - `def`
  - `theorem`
  - `lemma`
  - `structure` methods only if represented as standalone declarations in source
- Edit format in v1:
  - replace declaration body only
  - preserve declaration header/signature exactly
- Explicitly out of scope in v1:
  - cross-file edits
  - import changes
  - new file creation
  - deletions
  - whole-file rewrites

### 4.3 Search Engine

- Use MCTS over edit candidates
- One node = one cloned repo state plus metadata
- One edge = one single-function body replacement
- Root = clean clone from a pinned checkpoint/commit
- Depth limit = `1`
- Expansion uses the model as a prior over candidate edits
- Selection uses PUCT-style exploration
- Simulation = actual edit application + verification pipeline
- Backpropagation uses scalar reward only

### 4.4 Verification Gate

Mandatory checks, in order:

1. file parse succeeds
2. Lean type-check succeeds
3. targeted smoke/tests succeed
4. parity-sensitive checks still succeed where relevant

- Hard reject on any failure above
- Only surviving edits receive a nonzero fitness score

### 4.5 Fitness Function

Compile-first weighted reward:

`reward = gate * (base_pass_reward + metric_delta_reward + efficiency_reward)`

Where:

- `gate = 0` if parse/type-check/test/parity fails
- `gate = 1` otherwise

Initial reward terms:

- `base_pass_reward = +1.0` for passing the mandatory gate
- `metric_delta_reward`
  - `+5.0 * normalized_improvement` for improved target metric
  - `0` if unchanged within tolerance
  - negative if regression
- `efficiency_reward`
  - small penalty for slower execution or larger code size if the metric is tied

Primary target metric in v1:

- exact held-out next-byte accuracy on the fixed validation set for the byte-model backend

Secondary metrics:

- benchmark prompt score
- capped evaluation loss
- optional runtime improvement for selected Lean smoke paths

### 4.6 Acceptance Policy

- Do not auto-merge every positive edit
- For v1, accept only if:
  - the mandatory gate passed
  - reward exceeds the current baseline by a configured margin
  - no tracked metric regresses past tolerance
- Accepted edits are committed only to the clone branch, not to `main`
- Human review remains required before promoting an accepted edit to the primary branch

### 4.7 Logging and Reproducibility

Every rollout must persist:

- base checkpoint id
- repo commit hash
- candidate prompt
- edited declaration name
- original body hash
- candidate body hash
- verification results
- reward breakdown
- stdout/stderr for all checks
- timing
- model/checkpoint config used for proposal generation

### 4.8 Safety Statement

Refine the old "type-checking = safety" claim:

- accepted edits are guaranteed to be syntactically valid and type-correct under the enforced checks
- this does not imply semantic correctness, usefulness, or global safety
- all semantic claims require explicit evaluation metrics

### 4.9 V1 System Design

#### Planned folders and files

```text
self_edit/
  README.md
  schemas.py
  corpus.py
  declarations.py
  prompts.py
  apply_edit.py
  verify.py
  reward.py
  mcts.py
  runner.py
  reports.py

python_backend/scripts/
  build_edit_dataset.py
  train_edit_policy.py
  propose_edit.py
  run_self_edit_mcts.py
  evaluate_edit_policy.py
```

Lean-side additions later:

```text
MicroGPT.lean
  new CLI subcommands for:
  - config-aware checkpoint eval already exists
  - later add targeted benchmark eval hooks for self-edit reward
```

#### Public interfaces and types

```python
@dataclass(frozen=True)
class EditTarget:
    file_path: str
    decl_name: str
    decl_kind: Literal["def", "theorem", "lemma"]
    start_line: int
    end_line: int
    header_text: str
    body_text: str
```

```python
@dataclass(frozen=True)
class EditProposal:
    target: EditTarget
    prompt: str
    proposed_body: str
    temperature: float
    top_k: int
    source_checkpoint: str
    logprob_score: float | None
```

```python
@dataclass(frozen=True)
class VerificationResult:
    parse_ok: bool
    lean_ok: bool
    tests_ok: bool
    parity_ok: bool
    metric_results: dict[str, float]
    wall_time_sec: float
    stdout_path: str
    stderr_path: str
```

```python
@dataclass(frozen=True)
class RewardResult:
    total: float
    gate_passed: bool
    base_pass_reward: float
    metric_delta_reward: float
    efficiency_reward: float
    penalties: dict[str, float]
```

```python
@dataclass
class SearchNode:
    repo_dir: str
    parent_id: str | None
    proposal: EditProposal | None
    verification: VerificationResult | None
    reward: RewardResult | None
    visits: int
    value_sum: float
    children: list[str]
    is_terminal: bool
```

Output artifacts per run:

- `run.json`
- `nodes.jsonl`
- `accepted_edits.jsonl`
- `rejected_edits.jsonl`
- per-node logs under `artifacts/node_<id>/`

Use JSON lines for append-only observability.

#### Data flow

Stage 1: declaration extraction

- Build a declaration dataset from:
  - `MicroGPT.lean`
  - curated local Lean snippets already used in the byte corpus
  - optional additional small Lean files later
- Extraction rules:
  - use textual declaration detection first
  - keep only declarations under a configured max line count, default `80`
  - exclude declarations containing `sorry`
  - exclude declarations requiring import edits
  - store exact source spans

Stage 2: edit-policy prompt format

- Prompt the model with:
  - file path
  - declaration header
  - surrounding local context
  - instruction to emit replacement body only

Exact v1 prompt contract:

```text
You are editing one Lean declaration body.

File: <path>
Declaration: <decl_name>
Header:
<header_text>

Current body:
<body_text>

Task:
Produce a replacement body only.
Do not repeat the header.
Return plain Lean code for the body only.
```

Response parsing:

- accept raw body text only
- reject fenced code blocks
- reject outputs containing a new declaration header

Stage 3: edit application

- copy the repo into a fresh run-local clone directory
- replace only the target declaration body span
- write the modified file in the clone
- never mutate the source workspace in Phase 4 runs

Stage 4: verification

Run in order:

1. declaration-local parse sanity
2. `lean` type-check for the affected target path or project check command
3. targeted smoke/test command set
4. parity-sensitive Python/Lean checks if the edited area touches model semantics
5. optional benchmark eval if the gate passed

Stage 5: reward and search

- compute reward from verification outputs
- backpropagate reward through MCTS
- select the best child at the root
- for v1 depth-1 mode, this is effectively best-of-N verified proposals with MCTS accounting

#### Verification commands

Always-run gate:

- Lean type-check command for the project
- a fast smoke suite
- if Python backend files are touched:
  - Python syntax check
  - parity smoke for the relevant checkpoint/config

Metric tier, only if the gate passes:

- fixed held-out next-byte accuracy on the current best byte-model checkpoint
- benchmark prompt score
- optional runtime benchmark on selected commands

File-to-check mapping:

- if edit touches only `MicroGPT.lean`
  - run Lean type-check
  - run Lean smoke/parity benchmark hooks
- if edit touches only `python_backend/...`
  - run Python syntax
  - run parity check
  - run the selected eval metric
- if edit affects shared semantics or checkpoint format
  - run both Lean and Python checks

#### MCTS defaults

- Search algorithm: PUCT
- Rollout depth: `1`
- Root expansions per run: `32`
- Proposal count per target per expansion: `4`
- Temperature for proposal generation: `0.8`
- Top-k: `16`
- Value initialization for unseen nodes: `0`
- Exploration constant: `1.4`
- Terminal on:
  - invalid edit parse
  - failed verification gate
  - duplicate proposal body hash
  - timeout

For v1, treat MCTS as a disciplined ranking/search layer over verified single-edit proposals, not deep game-tree planning.

#### Candidate target selection

Do not let the search choose from the whole repo initially.

Use a maintained allowlist:

- `MicroGPT.lean`
- `python_backend/microgpt_torch/train.py`
- `python_backend/microgpt_torch/model.py`
- `python_backend/scripts/evaluate_next_token_accuracy.py`

Prioritize declaration targets by:

1. declarations already covered by smoke tests or parity checks
2. declarations under `80` lines
3. declarations with measurable downstream effect
4. declarations not edited in the last `N` accepted runs

Exclude for v1:

- checkpoint binary format code
- CLI parsing code in large mixed functions
- README/blueprint/docs
- declarations with heavy proof obligations unless explicitly targeted

#### Minimum proposer readiness criteria

Before real MCTS runs, require at least one checkpoint satisfying all:

- exact held-out next-byte accuracy >= `0.45` on the current validation slice
- benchmark prompt score exceeds the current `d_model=32` baseline
- sampled outputs on `def `, `theorem `, and `by ` contain recognizable Lean structure on manual inspection
- parity still within `1e-6` tolerance

Until then, self-modification runs are "sandbox research mode," not "candidate improvement mode."

#### Evaluation and acceptance criteria

Success criteria for Phase 4 v1:

1. declaration targets can be extracted deterministically from the repo
2. the model can generate at least one syntactically parseable replacement body for a target
3. the system can apply that edit to a clone without mutating the source repo
4. the full verification gate produces a scalar reward
5. MCTS can rank proposals and select a best verified edit
6. the run produces complete logs and artifacts
7. at least one accepted edit improves a tracked metric without failing parity or type-checking

Failure criteria:

- edits frequently break file structure because body-only replacement is underspecified
- verification takes too long to support iterative search
- reward is dominated by noise and fails to rank obviously better edits
- accepted edits improve one metric while silently regressing core parity or smoke checks

#### Test plan

Unit tests:

- declaration span extraction
- body-only replacement correctness
- duplicate proposal detection by body hash
- reward computation edge cases
- PUCT selection and backprop math

Integration tests:

- one declaration target in a temp clone
- one generated proposal
- full verification gate
- JSON artifact emission

Regression tests:

- parity still passes after accepted edits on model-related files
- failed edits never touch the source workspace
- repeated runs with the same seed produce the same candidate ordering where deterministic settings apply

Dry-run scenario:

- choose one safe target in `MicroGPT.lean`
- inject a known no-op or semantics-preserving body edit
- verify the system accepts it
- inject a known broken edit
- verify hard rejection

**Estimated effort:** Open research problem (months to years)
**Dependencies:** Phase 3.7, Phase 3.8, and the existing Lean parity/checkpoint bridge
**This is the paper.** Even partial progress here is publishable.

---

## Phase 5: Formal Verification (Ongoing)

**Goal:** Prove mathematical properties about the model in Lean itself.

```
├── 5.1 Provable Now (Type-Level)
│   ├── ✅ Dimension safety (already done via dependent types)
│   ├── ✅ No shape mismatches possible
│   └── ✅ Residual connections preserve dimensions
│
├── 5.2 Provable With Effort
│   ├── Status: not yet formally proved in Lean; currently treated as targets for future proof work
│   ├── softmax outputs sum to 1.0
│   ├── softmax outputs are non-negative
│   ├── relu output ≥ 0
│   ├── layer norm output has mean ≈ 0, variance ≈ 1
│   ├── attention weights sum to 1.0 per query
│   └── cross-entropy loss ≥ 0
│
├── 5.3 Research-Level Proofs
│   ├── Status: not yet formally proved; current evidence is empirical (finite differences, parity, and training behavior), not theorem-level
│   ├── Gradient computation is correct (AD correctness)
│   ├── Training decreases loss (under assumptions)
│   ├── Self-modifications preserve specified invariants
│   └── Convergence properties of training
│
└── 5.4 Lean-Specific Advantages
    ├── Proofs compose: prove each layer correct → whole model correct
    ├── Refactoring with confidence: change code, proofs break if wrong
    └── Proofs are checkable by machine, not just peer review
```

**Estimated effort:** Ongoing, parallel to other phases
**Dependency:** Can start immediately for basic properties

---

## Dependency Graph

```
Phase 0 (Done)
  │
  ▼
Phase 1: Complete Forward Pass ──────────► Phase 5: Verification
  │                                         (ongoing, parallel)
  ▼
Phase 2: Automatic Differentiation ◄─────── HARDEST PHASE
  │
  ▼
Phase 3: Training Loop
  │
  ▼
Phase 3.8: Edit-Policy Preparation
  │
  ▼
Phase 4: Self-Modification ◄──────────────── THE PAPER
```

---

## Minimum Publishable Results

Each checkpoint below is independently publishable/shareable:

| Checkpoint | What You Have | Audience |
|------------|--------------|----------|
| Phase 1 complete | "Formally typed transformer inference in Lean 4" | Lean community, Twitter/X |
| Phase 2 complete | "Reverse-mode autodiff in Lean 4" | PL research, ML research |
| Phase 3 complete | "End-to-end transformer training in Lean 4" | Major interest, blog post worthy |
| Phase 4 partial | "Self-modifying neural net with formal verification" | Conference paper territory |
| Phase 4 complete | "Verified self-improving AI" | Top venue paper |

---

## Public Repo Readiness

This project is also a portfolio/research artifact, not just an internal prototype.

```
├── README.md
│   ├── Explain the project in plain language
│   ├── Show what is already implemented vs planned
│   ├── Include exact commands to type-check, run smoke tests, and run native experiments
│   ├── Document the Lean-spec / PyTorch-compute split honestly
│   ├── Explain why future self-modification uses clone-only workspaces
│   └── State the current safety boundaries and acceptance metrics clearly
│
├── Public GitHub Repository
│   ├── Treat the repo as a credibility asset for ML, PL, and AI safety work
│   ├── Make the current verified state legible to researchers and engineers
│   ├── Keep roadmap and limitations explicit so the repo does not overclaim
│   └── Invite focused outside contributions from the Lean community
│
└── Contribution Surface
    ├── Bug fixes in tensor/AD/training code
    ├── Proof work in Phase 5
    ├── PyTorch parity backend and checkpoint bridge
    └── Better tooling around experiments, checkpoints, and dataset curation
```

---

## Tech Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Lean version | 4.x (latest stable) | Lean 3 is deprecated |
| Float type | Native Float (IEEE 754) | Lean's built-in, fast |
| Tokenization | Byte-level (char) | Simplest, no BPE needed |
| Architecture | GPT-2 style (pre-norm) | Simple, well-understood |
| Training data | Mathlib + own source | On-theme, freely available |
| Batch size | 1 (start) | CPU-bound anyway |
| Context length | 128-256 tokens | Enough for Lean snippets |
| Model size | d_model=64, 2 layers | Trainable on CPU |

---

## Development Tools

Use the validation stack that is already part of the project, rather than depending on an external symbolic system.

### Validation and Development Stack

| Tool | Purpose |
|------|---------|
| Lean 4 | Canonical implementation, type-level guarantees, smoke tests, and checkpoint verification |
| Finite differences | Validate autodiff rules numerically at the operation and model-block level |
| Python + PyTorch | Reference implementation, fast training backend, and parity target |
| Lean↔PyTorch parity checks | Verify matching logits/loss for the same weights and inputs |
| JSONL training logs | Track loss, lr, grad norm, and throughput across experiments |
| Git | Version control for accepted experiments and later accepted self-modifications |

### Recommended workflow

```
1. Implement or change a numerical rule   → Lean and/or PyTorch
2. Validate gradients numerically         → finite differences
3. Compare Lean vs PyTorch outputs        → parity checks
4. Train on a controlled byte task        → PyTorch
5. Re-load checkpoint into Lean           → checkpoint + parity verification
6. Only then scale experiments further
```

### Why this is enough for now

- Lean already enforces tensor-shape and type correctness.
- Finite-difference checks catch incorrect local gradient rules.
- Lean↔PyTorch parity checks catch implementation drift between the spec and the training backend.
- End-to-end checkpoint round-tripping proves the practical bridge really works.
- Full formal proofs of numerical properties remain Phase 5 work, not a prerequisite for useful experiments.

---

## What NOT To Build

Explicitly out of scope to keep focus:

- Multi-GPU / distributed training
- BPE tokenizer
- Flash attention or memory optimizations
- Web UI or API server
- Comparison benchmarks against PyTorch
- RLHF or any alignment techniques
- Production deployment concerns

Build the smallest thing that demonstrates the core idea:
**a neural network that modifies itself and proves its modifications are safe.**
