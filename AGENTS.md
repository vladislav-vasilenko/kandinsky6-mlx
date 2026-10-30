# Rules for executing a stage spec

Each file in `specs/` is executed by one run. The run reads the spec, does the work, proves it with tests and validation runs, and writes a report. These rules apply to every spec.

## Ground truth

- The reference is `diffusers==0.42.1` with `transformers==5.22.1`, `accelerate==1.13.0`, run by PyTorch on CPU in float32 unless the spec says otherwise.
- Read the reference source before porting it. Sources are cited as `diffusers/<path>:<lines>` for version 0.42.1. If installed line numbers differ, find the same class or function by name.
- Never change reference behavior to make a test pass. A port matches the reference or the difference is documented and justified in the report.

## Environment

- Python is managed with `uv`. Use `uv run` for every command; do not install into the system interpreter.
- MLX runs only on Apple Silicon. If `import mlx.core` fails, stop and report `BLOCKED: mlx unavailable`.
- Real checkpoints are large (Lite media part about 9.4 GiB, Qwen 15.5 GiB, Pro DiT about 60 GB). A spec stage that needs real weights first checks free disk space and the presence of weights. If either is missing, it completes the tiny-fixture part, marks the real-weight part `SKIPPED` with the reason, and does not download anything on its own.
- PyTorch reference runs on real weights may be done on Kaggle. MLX runs need a Mac.

## Determinism

- MLX and PyTorch random generators differ. Parity tests always generate noise and inputs with NumPy (`np.random.default_rng(seed)`) or load them from fixtures, and pass the same arrays to both implementations.
- Pipelines accept explicit initial latents so the MLX and diffusers runs start from the same noise.

## Precision

- Checkpoint DiT weights are BF16 with some F32 tensors; VAE, audio VAE, vocoder and CLIP are F32.
- Diffusers keeps `time_embeddings` and every `*modulation*` module of the DiT in float32 (`_keep_in_fp32_modules`), and computes `apply_scale_shift`, `apply_gate_sum` and RoPE in float32. The MLX port follows the same policy.
- Parity is checked first in float32 against float32, then in bf16 against the float32 reference.

## Tolerances

Use `kandinsky6_mlx.utils.compare` (created in spec 01) for all comparisons. It reports `max_abs`, `rel_max = max_abs / max(|ref|)`, `rel_l2` and `cosine`.

| Comparison | Default criterion |
|---|---|
| float32 MLX vs float32 reference, single primitive | `rel_max <= 1e-5` |
| float32 MLX vs float32 reference, block or tiny model | `rel_max <= 1e-4` |
| bf16 MLX vs float32 reference | `rel_l2 <= 2 * rel_l2(torch bf16 vs torch fp32)` on the same inputs, and `cosine >= 0.999` |

A spec may tighten or relax a tolerance; relaxing requires a written reason in the report.

## Definition of done

1. Every deliverable listed in the spec exists.
2. `uv run pytest` passes for the tests the spec adds and for all earlier tests.
3. `uv run ruff check` passes.
4. The validation runs listed in the spec were executed and their numbers are in the report.
5. `reports/<spec-id>.json` is written with the schema below.

## Report schema

```json
{
  "spec": "05-dit-model",
  "status": "PASS | PARTIAL | BLOCKED | FAIL",
  "date": "YYYY-MM-DD",
  "environment": {"chip": "...", "macos": "...", "mlx": "...", "torch": "...", "diffusers": "0.42.1"},
  "deliverables": ["paths created or changed"],
  "tests": {"command": "...", "passed": 0, "failed": 0},
  "validation": [{"name": "...", "metrics": {}, "criterion": "...", "pass": true}],
  "skipped": [{"what": "...", "reason": "..."}],
  "deviations": [{"what": "...", "why": "..."}],
  "follow_ups": ["..."]
}
```

`PARTIAL` means the tiny-fixture part passed and a real-weight part was skipped for an environment reason. `FAIL` means a criterion was run and not met.

## Attribution

The repository owner is the only author. Do not credit any AI assistant or model anywhere:

- no `Co-Authored-By:` or similar trailers naming an assistant or model in commit messages;
- no "Generated with …" footers or assistant mentions in pull requests, issues, release notes, model cards, READMEs or code comments;
- do not add an assistant or model to author, contributor or maintainer lists (`pyproject.toml`, `CITATION.cff`, model cards).

## Scope discipline

- Do only what the spec asks. Note anything else under `follow_ups`.
- Do not change public APIs created by earlier specs without updating their tests and noting it under `deviations`.
- Keep commits focused on one spec.
