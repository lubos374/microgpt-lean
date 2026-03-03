SHELL := /bin/bash
PYTHON := $(shell if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

.PHONY: check smoke parity pycheck test

check:
	@echo "==> Lean type-check"
	@lean MicroGPT.lean

smoke:
	@echo "==> Lean smoke suite"
	@lean --run MicroGPT.lean

parity:
	@echo "==> Lean↔PyTorch parity"
	@$(PYTHON) python_backend/scripts/parity_check.py --fixture byte --checkpoint python_backend/checkpoints/latest.bin

pycheck:
	@echo "==> Python syntax check"
	@files=$$(find python_backend self_edit -name '*.py' -type f | sort); \
	$(PYTHON) -m py_compile $$files

test:
	@$(MAKE) check
	@$(MAKE) smoke
	@$(MAKE) pycheck
	@$(MAKE) parity
