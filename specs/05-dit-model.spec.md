---
id: 05-dit-model
title: Full Kandinsky6Transformer3DModel for Lite and Pro configs
depends_on: [04-dit-blocks]
estimate: 2 days
real_weights: stage-b
---

## Goal

The complete DiT that loads converted weights for any Kandinsky 6 checkpoint from its `config.json`, with float32 parity on tiny fixtures and per-block bf16 parity on the real Lite-distill weights.

## Read first

- `transformer_kandinsky6.py:591-844` `Kandinsky6Transformer3DModel` (constructor, `forward`).
- Real configs (all share flags `ca_rope`, `cross_gates`, `fix_modulation`, `visual_cond` = True, `visual_token_type_num_embeddings=2`, `in_visual_dim=16`):
  - Lite: `model_dim=1792, num_visual_blocks=32, num_text_blocks=2, ff_dim=7168, model_dim_a=896, ff_dim_a=3584, axes_dims=[16,24,24], time_dim=512, out_visual_dim=160, out_audio_dim=400, in_audio_dim=40, audio_freqs_scaling=0.144, scale_factor=[1,2,2]`.
  - Pro: `model_dim=4096, num_visual_blocks=60, num_text_blocks=4, ff_dim=16384, model_dim_a=2048, ff_dim_a=7168, axes_dims=[32,48,48], time_dim=1024`.
- The checkpoint config may contain keys the reference ignores (e.g. `is_multimodal`); ignore them the same way.

## Deliverables

- `src/kandinsky6_mlx/models/dit/model.py` with `Kandinsky6Transformer3DModel.from_converted(path, dtype)`.
- `tests/parity/test_dit_model.py` (stage A), `tests/parity/test_dit_real.py` (stage B, marker `real_weights`).
- `reference/record_dit_real.py` (uses the spec 01 recorder) and instructions in its docstring for running it on Kaggle.
- `reports/05-dit-model.json`.

## Implementation notes

- Forward order: text RoPE from `arange(L_text)` → two text branches (video and audio, each with its own embeddings, time embedding + pooled embedding, encoder blocks) → visual patchify (+ token type embeddings per frame) → visual RoPE from per-axis positions (default `arange`, overridable by `visual_rope_pos`) → flatten → audio embeddings and 1D RoPE → fused blocks → reshape to `(B, T', H', W', C)` → output heads.
- The timestep passed to the model is on the `[0, num_train_timesteps]` scale; check how the pipeline passes it in spec 09.
- Memory: build the model with `mx` lazy evaluation and load weights before any forward; for stage B use the Lite model only.

## Tests

- Stage A: tiny model float32 parity of final video and audio outputs and of every block output (`rel_max <= 1e-4`); with and without audio; with custom `visual_rope_pos` and `visual_token_type_ids`.
- Config loading: constructing from the real Lite and Pro `config.json` files (no weights) yields the expected parameter count and names (compare with the reference model built on the `meta` device).
- Stage B: on real Lite-distill weights and recorded Smoke-size inputs (latents `[1, 16, 7, 32, 54]` before patchify → 3024 video tokens, audio `[1, 45, 40]`, real cached text embeddings, a mid-trajectory timestep), compare each block output and the final outputs in bf16 against the float32 reference.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_dit_model.py -q
# Stage B (weights and recorded activations present):
uv run pytest tests/parity/test_dit_real.py -m real_weights -q
```

Report per-block `rel_l2` and `cosine` for stage B, and the torch bf16-vs-fp32 drift used as the tolerance basis.

## Acceptance criteria

- [ ] Stage A passes at `rel_max <= 1e-4`.
- [ ] Lite and Pro configs construct with exactly the reference parameter names and shapes.
- [ ] Stage B (if run): every block and the final outputs satisfy the bf16 criterion from `AGENTS.md`; the first block exceeding it is named in the report if not.

## Out of scope

Sampling loop, VAEs, Pro real-weight runs (spec 14 decides whether Pro is feasible).

## Stop conditions

Stage B error grows steadily across blocks beyond the criterion → `FAIL`, attach the per-block curve; do not tune tolerances to pass.
