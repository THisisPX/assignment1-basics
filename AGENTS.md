# CS336 Spring 2025 Assignment 1: Basics

This is a Stanford CS336 course assignment for implementing foundational large language model (LLM) components from scratch in PyTorch. The project uses `uv` for Python environment and package management, and `pytest` with snapshot testing for validation.

## Technology Stack

- **Python**: >=3.11
- **Deep Learning**: PyTorch (~2.6.0, ~2.2.2 on Intel macOS)
- **Environment & Packaging**: `uv` (managed virtual environment in `.venv/`)
- **Build Backend**: `uv_build` (configured in `pyproject.toml`)
- **Testing**: `pytest>=8.3.4`
- **Tensor Utilities**: `einops>=0.8.1`, `einx>=0.3.0`, `jaxtyping>=0.3.0`
- **Tokenizer References**: `tiktoken>=0.9.0`
- **Experiment Tracking**: `wandb>=0.19.7`
- **Cluster Job Submission**: `submitit>=1.5.2`
- **Linting**: `ruff` (line-length 120, configured in `pyproject.toml`)
- **Other**: `numpy`, `regex>=2024.11.6`, `tqdm>=4.67.1`, `psutil>=6.1.1`

Note: The `pyproject.toml` configures a Tsinghua University PyPI mirror as the default index.

## Project Structure

```
cs336_basics/               # Main Python package (student implementations)
  __init__.py               # Package version from metadata
  Tokenizar.py              # BPE Tokenizer class (encode/decode)
  train_bpe.py              # BPE training algorithm (multiprocessing + heap-based)
  pretokenization_example.py# Example: chunking files for parallel pretokenization

tests/                      # Test suite and adapters
  adapters.py               # Bridge functions connecting tests to implementations.
                            # Each function either raises NotImplementedError or imports
                            # from cs336_basics. This is the primary file students modify
                            # to wire their code into the test suite.
  conftest.py               # pytest fixtures: NumpySnapshot, Snapshot, model params,
                            # random tensors, and ts_state_dict (reference model weights)
  common.py                 # Shared utilities: FIXTURES_PATH, gpt2_bytes_to_unicode()
  test_model.py             # Tests for Linear, Embedding, SwiGLU, SDPA, MHA, RoPE,
                            # TransformerBlock, TransformerLM
  test_nn_utils.py          # Tests for RMSNorm, SiLU, softmax, cross-entropy, grad clipping
  test_data.py              # Tests for get_batch (language modeling data sampling)
  test_optimizer.py         # Tests for AdamW and cosine LR schedule
  test_serialization.py     # Tests for checkpoint save/load
  test_tokenizer.py         # Tests for BPE tokenizer vs tiktoken (roundtrip, memory)
  test_train_bpe.py         # Tests for BPE training speed, correctness, special tokens
  fixtures/                 # Test data: GPT-2 vocab/merges, sample corpora,
                            # reference BPE outputs, TinyStories model weights
  _snapshots/               # Stored numerical outputs (.npz) and pickle snapshots (.pkl)

cs336/                      # Empty directory (reserved for course infrastructure)
```

## Build and Test Commands

All commands are run through `uv`, which automatically manages the virtual environment.

```bash
# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/test_model.py

# Run with verbose output
uv run pytest -v

# Run with snapshot exact matching (no tolerance)
uv run pytest --snapshot-exact

# Run a single Python script in the managed environment
uv run <path/to/script.py>
```

### Linting

Ruff is configured in `pyproject.toml` with a line-length of 120. There are per-file ignores for `__init__.py` (E402, F401, F403, E501).

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .
```

### Creating a Submission

```bash
./make_submission.sh
```

This runs the full test suite (outputting `test_results.xml`) and creates `cs336-spring2025-assignment-1-submission.zip` while excluding build artifacts, caches, fixtures, snapshots, logs, and the `.git` / `.venv` directories.

## Code Organization and Module Divisions

The assignment is divided into functional areas, each exercised by a corresponding test file:

1. **Neural Network Primitives** (`test_nn_utils.py`)
   - `run_rmsnorm`, `run_silu`, `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`

2. **Model Layers** (`test_model.py`)
   - `run_linear`, `run_embedding`, `run_swiglu`
   - `run_scaled_dot_product_attention`
   - `run_multihead_self_attention` (without RoPE)
   - `run_multihead_self_attention_with_rope`
   - `run_rope`
   - `run_transformer_block`
   - `run_transformer_lm`

3. **Data Utilities** (`test_data.py`)
   - `run_get_batch`: Sample (input, label) pairs from a 1D token-id array for language modeling.

4. **Optimization** (`test_optimizer.py`)
   - `get_adamw_cls`: Return a custom `torch.optim.Optimizer` implementing AdamW.
   - `run_get_lr_cosine_schedule`: Cosine annealing with linear warmup.

5. **Serialization** (`test_serialization.py`)
   - `run_save_checkpoint`, `run_load_checkpoint`: Serialize / deserialize model, optimizer, and iteration counter.

6. **Tokenizer** (`test_tokenizer.py`)
   - `get_tokenizer`: Return a BPE tokenizer instance given vocab, merges, and special tokens.

7. **BPE Training** (`test_train_bpe.py`)
   - `run_train_bpe`: Train a BPE tokenizer from a corpus file, outputting vocab and merges.

### Adapter Pattern

The test suite does **not** import student modules directly. Instead, it imports from `tests/adapters.py`. Each adapter function is a thin wrapper that either:

- Raises `NotImplementedError` (the initial state), or
- Imports and calls the student's implementation from `cs336_basics`.

**Convention for agents**: When implementing a component, place the core logic in `cs336_basics/` and wire it into `tests/adapters.py` by replacing the `raise NotImplementedError` with an actual call.

## Testing Strategy

### Snapshot Testing

Many model tests use numerical snapshot testing via custom fixtures (`numpy_snapshot`, `snapshot`):

- `NumpySnapshot` stores expected outputs as `.npz` files in `tests/_snapshots/`.
- `Snapshot` stores arbitrary data as `.pkl` files.
- Tests compare the student's output against the stored snapshot using `np.testing.assert_allclose` with tolerances (`rtol=1e-4`, `atol=1e-2` by default; some tests use stricter values like `atol=1e-6`).

### Exact Equality Testing

Tokenizer tests compare outputs against `tiktoken.get_encoding("gpt2")` with exact list equality (`==`). Roundtrip tests assert that `decode(encode(text)) == text`.

### Performance and Resource Testing

- `test_train_bpe_speed`: BPE training on a small reference corpus must complete in under 1.5 seconds.
- `test_encode_iterable_memory_usage` / `test_encode_memory_usage`: Memory-constrained tests (Linux only, uses `resource.setrlimit`) verify that `encode_iterable` is memory-efficient. These are skipped on non-Linux platforms.

### Randomness and Fixtures

`conftest.py` seeds PyTorch manual seeds deterministically for fixtures (`q`, `k`, `v`, `in_embeddings`, `mask`, `in_indices`). This ensures reproducible snapshot comparisons.

## Development Conventions

- **Type Hints**: Heavy use of `jaxtyping` tensor annotations (e.g., `Float[Tensor, "batch seq d_model"]`) and standard Python typing.
- **Line Length**: 120 characters (Ruff configuration).
- **File Encodings**: UTF-8.
- **Special Tokens**: BPE implementations must preserve special tokens as atomic units (e.g., `<|endoftext|>`) and not merge them with adjacent text.
- **Tiebreaking in BPE**: When frequencies are equal, prefer the lexicographically greatest byte pair (tuple comparison on the byte representations). The reference implementation uses GPT-2's byte-to-unicode remapping only for serialization, not for tiebreaking.
- **Parallel Pretokenization**: The assignment encourages multiprocessing for BPE pretokenization. `pretokenization_example.py` provides a reference pattern for finding chunk boundaries in a file to split work across processes.

## Data Setup

The README provides commands to download TinyStories and an OpenWebText sample into a `data/` directory. These are used for the training experiments described in the assignment PDF, not for the unit tests (which use `tests/fixtures/`).

```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

## Cluster / SLURM Support

A sample SLURM script (`train_test.slurm`) is present. It sets ulimits, loads CUDA/cuDNN modules, activates the `.venv`, and runs `uv run pytest`. The project depends on `submitit` for programmatic SLURM job submission.

## Security Considerations

- The project runs student-provided Python code and executes it via `pytest`. There is no sandboxing.
- The `uv` virtual environment is local to the project (`./.venv`). No system-wide package installation is required.
- `make_submission.sh` excludes sensitive artifacts (`.env`, `.git`, `.venv`, logs, large data files) from the submission zip, but agents should avoid placing secrets in the working directory.
