# Usage

This file collects the longer command cookbook. The main README only keeps the
short public quickstart.

## Lean

Type-check the Lean reference implementation:

```bash
lean MicroGPT.lean
```

Run the full smoke suite:

```bash
lean --run MicroGPT.lean
```

Print the built-in parity default case:

```bash
lean --run MicroGPT.lean parity-default-case tiny-single-head
```

Export a Lean fixture checkpoint:

```bash
lean --run MicroGPT.lean parity-save tiny-single-head /tmp/tiny-single-head.bin
```

Evaluate Lean logits and loss for a fixture checkpoint:

```bash
lean --run MicroGPT.lean parity-eval tiny-single-head /tmp/tiny-single-head.bin 0,1 1 0
```

## PyTorch parity and training

Run the default public parity check:

```bash
python3 python_backend/scripts/parity_check.py --fixture byte --checkpoint python_backend/checkpoints/latest.bin
```

Run a parity check with an explicit config sidecar:

```bash
python3 python_backend/scripts/parity_check.py \
  --config-json python_backend/checkpoints/run-20000-d32-ctx16/config.json \
  --checkpoint python_backend/checkpoints/run-20000-d32-ctx16/best.bin \
  --tokens 100,101,102,32,58,32,70,108,111,97,116,10,32,32,120,32 \
  --predict-idx 15 \
  --target 58
```

Train the byte-level PyTorch mirror:

```bash
python3 python_backend/scripts/train_byte_model.py \
  --steps 100 \
  --batch-size 32 \
  --checkpoint-dir python_backend/checkpoints
```

Train and run a parity smoke check on one saved checkpoint:

```bash
python3 python_backend/scripts/train_byte_model.py \
  --steps 100 \
  --batch-size 32 \
  --checkpoint-dir python_backend/checkpoints \
  --parity-smoke
```

Resume a run from an earlier checkpoint:

```bash
python3 python_backend/scripts/train_byte_model.py \
  --resume-from python_backend/checkpoints/run-0500/latest.bin \
  --steps 500 \
  --log-every 50 \
  --checkpoint-every 250 \
  --warmup-steps 50 \
  --grad-clip-norm 1.0 \
  --train-eval-limit 4096 \
  --valid-eval-limit 4096 \
  --checkpoint-dir python_backend/checkpoints/run-1000 \
  --best-on-valid-loss \
  --sample-after \
  --sample-greedy
```

Sample from a saved checkpoint:

```bash
python3 python_backend/scripts/sample_checkpoint.py \
  --checkpoint python_backend/checkpoints/latest.bin \
  --prompt "def " \
  --prompt "theorem " \
  --prompt "by " \
  --greedy
```

Evaluate multiple checkpoints:

```bash
python3 python_backend/scripts/evaluate_checkpoints.py \
  --checkpoint-dir python_backend/checkpoints/run-0500 \
  --glob "step-*.bin" \
  --parity
```

Measure exact next-byte loss and accuracy:

```bash
python3 python_backend/scripts/evaluate_next_token_accuracy.py \
  --checkpoint python_backend/checkpoints/run-5000-ctx16/latest.bin \
  --dataset priority-valid \
  --context 16 \
  --limit 4096
```

Select the best checkpoint by exact held-out accuracy:

```bash
python3 python_backend/scripts/evaluate_next_token_accuracy.py \
  --checkpoint-dir python_backend/checkpoints/run-20000-d32-ctx16 \
  --glob "step-*.bin" \
  --dataset priority-valid \
  --context 16 \
  --limit 4096 \
  --metric accuracy \
  --select-best-to python_backend/checkpoints/run-20000-d32-ctx16/best-accuracy.bin
```

## Edit-policy workflow (experimental)

Everything in this section is experimental and is not part of the core
verified release claim.

Build the declaration-level edit dataset:

```bash
python3 python_backend/scripts/build_edit_dataset.py \
  --output python_backend/edit_data/edit_dataset.jsonl
```

Evaluate a checkpoint as an edit proposer:

```bash
python3 python_backend/scripts/evaluate_edit_policy.py \
  --checkpoint python_backend/checkpoints/run-20000-d32-l2-ctx16/best-accuracy.bin \
  --config-json python_backend/checkpoints/run-20000-d32-l2-ctx16/config.json \
  --dataset python_backend/edit_data/edit_dataset.jsonl \
  --split valid \
  --context 16
```

Generate one replacement-body proposal:

```bash
python3 python_backend/scripts/propose_edit.py \
  --checkpoint python_backend/checkpoints/run-20000-d32-l2-ctx16/best-accuracy.bin \
  --config-json python_backend/checkpoints/run-20000-d32-l2-ctx16/config.json \
  --decl-name Vec.ofFn \
  --context 16
```

Sample multiple stochastic replacement-body candidates:

```bash
python3 python_backend/scripts/propose_edit.py \
  --checkpoint python_backend/checkpoints/run-edit-policy-d32-l2-ctx16-v3/best.bin \
  --config-json python_backend/checkpoints/run-edit-policy-d32-l2-ctx16-v3/config.json \
  --decl-name transformerForward \
  --context 16 \
  --max-new-tokens 80 \
  --temperature 0.9 \
  --top-k 16 \
  --num-samples 5
```

Fine-tune the current best checkpoint on edit prompt/response pairs:

```bash
python3 python_backend/scripts/train_edit_policy.py \
  --checkpoint python_backend/checkpoints/run-20000-d32-l2-ctx16/best-accuracy.bin \
  --config-json python_backend/checkpoints/run-20000-d32-l2-ctx16/config.json \
  --dataset python_backend/edit_data/edit_dataset.jsonl \
  --context 16 \
  --epochs 5 \
  --checkpoint-dir python_backend/checkpoints/edit-policy-d32-l2-ctx16
```
