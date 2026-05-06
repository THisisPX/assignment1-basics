# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stanford CS336 course assignment implementing foundational LLM components from scratch in PyTorch. Students implement neural network primitives, attention mechanisms, BPE tokenization, and training utilities.

## Common Commands

```bash
# Run full test suite
uv run pytest

# Run specific test file
uv run pytest tests/test_model.py

# Run with verbose output
uv run pytest -v

# Run with exact snapshot matching (no tolerance)
uv run pytest --snapshot-exact

# Lint code
uv run ruff check .
uv run ruff format .

# Create submission
./make_submission.sh
```

## Architecture

### Module Structure

- **`cs336_basics/transformer.py`** — Neural network components: `Linear`, `Embedding`, `RMSnorm`, `FFN` (SwiGLU), `RotaryEmbedding`, `multihead_self_attention`. Uses `einops`/`einx` for tensor operations and `jaxtyping` for typed tensor annotations.
- **`cs336_basics/Tokenizar.py`** — BPE tokenizer (`BPE_Tokenizer`) with encode/decode and special token handling.
- **`cs336_basics/train_bpe.py`** — BPE training algorithm (multiprocessing + heap-based).
- **`tests/adapters.py`** — Bridge between tests and student implementations. Each function either raises `NotImplementedError` (initial state) or delegates to `cs336_basics`. This is the primary wiring file students modify.

### Adapter Pattern

Tests do NOT import from `cs336_basics` directly. They import from `tests/adapters.py`, which maps test functions to student implementations. To wire in your code, replace `raise NotImplementedError` in `adapters.py` with a call to your implementation.

### Testing Strategy

- **Snapshot testing**: Model outputs compared against `.npz`/`.pkl` files in `tests/_snapshots/` using `np.testing.assert_allclose` (tolerances vary by test).
- **Exact equality**: Tokenizer tests compare against `tiktoken.get_encoding("gpt2")` with list equality.
- **Performance testing**: BPE training speed must complete within 1.5s on reference corpus.
- **`conftest.py`** seeds PyTorch deterministically for reproducibility.

### BPE Tiebreaking Rule

When frequencies are equal, prefer the lexicographically greatest byte pair (comparing byte tuples). The GPT-2 byte-to-unicode remapping is used only for serialization, not for tiebreaking.

## Key Implementation Notes

- Heavy use of `jaxtyping` tensor annotations (e.g., `Float[Tensor, "batch seq d_model"]`)
- `RotaryEmbedding` caches frequency computed tensors for efficiency
- SwiGLU activation: `GLU(x) = sigmoid(W1x) * W3x`, output = `GLU(W1x) * W2`
- RMSNorm: `output = x * g / sqrt(mean(x^2) + eps)`
- Line length: 120 characters (Ruff configured in `pyproject.toml`)