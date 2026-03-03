# Experimental

Experimental. Not part of the core verified release claim.

## Self-Edit Data Prep

This folder contains the first `Phase 3.8` building blocks for turning the
current byte-level model into an edit proposer.

Implemented here:

- declaration-level corpus loading from `MicroGPT.lean` plus curated Lean snippets
- simple top-level `def` / `theorem` / `lemma` extraction
- body-only edit prompt formatting
- JSONL dataset generation for train/validation splits

Current scope is intentionally narrow:

- textual extraction only
- only declarations with an explicit `:=` body are included
- declarations containing `sorry` are excluded
- large declarations can be filtered by line count

This is dataset-preparation infrastructure, not the self-modification loop
itself.
