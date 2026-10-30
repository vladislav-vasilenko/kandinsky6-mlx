---
id: 00-bootstrap
title: Project tooling, lockfile and CPU test run
depends_on: []
estimate: 0.5 days
real_weights: none
---

## Goal

A clean project that installs with `uv`, imports `mlx`, runs an empty test suite and lint on Apple Silicon, and has the reference group (PyTorch, diffusers 0.42.1) installable separately.

## Read first

- `README.md`, `AGENTS.md`, `specs/README.md` in this repository.
- `pyproject.toml` (already drafted; adjust versions only if resolution fails).

## Deliverables

- `uv.lock` created by `uv lock`.
- `tests/unit/test_environment.py`.
- `scripts/check_environment.py` that prints chip, macOS, Python, MLX, NumPy, safetensors and, if the reference group is installed, torch/diffusers/transformers versions, plus free disk space in the repository volume. Exit code 0.
- `.github/workflows/ci.yml`: macOS arm64 runner, `uv sync --group dev`, `uv run ruff check`, `uv run pytest -m "not real_weights and not slow"`.
- `reports/00-bootstrap.json`.

## Implementation notes

- Keep the reference group optional: the default install must not pull PyTorch.
- `diffusers==0.42.1`, `transformers==5.22.1`, `accelerate==1.13.0` are fixed; they are the versions the port is validated against.
- `transformers 5.22.1` requires `huggingface_hub<2`; keep that bound.

## Tests

- `test_mlx_available`: `mx.array([1.0]) + 1` evaluates on the default device.
- `test_bf16_supported`: a bf16 matmul evaluates and returns dtype `bfloat16`.
- `test_reference_group_optional`: importing `kandinsky6_mlx` does not import `torch`.

## Validation runs

```bash
uv sync --group dev
uv run python scripts/check_environment.py
uv run pytest -q
uv run ruff check
uv sync --group dev --group reference
uv run python -c "import diffusers, torch; print(diffusers.__version__, torch.__version__)"
```

Record all versions and the free disk space in the report.

## Acceptance criteria

- [ ] `uv sync --group dev` and `uv sync --group dev --group reference` both succeed.
- [ ] `uv run pytest -q` passes; `uv run ruff check` passes.
- [ ] `import kandinsky6_mlx` works without the reference group installed.
- [ ] The report contains environment versions and free disk space.

## Out of scope

Any model code. Downloading checkpoints.

## Stop conditions

`import mlx.core` fails on the machine → `BLOCKED: mlx unavailable`.
