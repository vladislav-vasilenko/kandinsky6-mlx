---
id: 01-reference-harness
title: Comparison utilities, fixture format and tiny reference fixtures
depends_on: [00-bootstrap]
estimate: 1.5 days
real_weights: none (real-fixture runner is written here, executed in later specs)
---

## Goal

Every later spec compares MLX output with diffusers output. This spec builds that machinery once: metrics, a fixture file format, tiny random-weight reference models, and a runner that records activations of real models so that real-weight parity can be checked on a Mac without running PyTorch there.

## Read first

- `diffusers/models/transformers/transformer_kandinsky6.py` (whole file, 844 lines): class names and constructor arguments.
- `diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py:625-` `AutoencoderKLHunyuanVideo.__init__`.
- `diffusers/models/autoencoders/autoencoder_mmaudio.py:350-` `MMAudioVAE`.
- `diffusers/models/autoencoders/mmaudio_vocoder.py:154-` `MMAudioVocoder`.
- `diffusers/schedulers/scheduling_piflow.py`.

## Deliverables

- `src/kandinsky6_mlx/utils/compare.py`: `compare(actual, reference) -> dict` with `max_abs`, `rel_max`, `rel_l2`, `cosine`, `shape`, `dtype`; accepts MLX arrays, NumPy arrays and torch tensors; computes in float64.
- `src/kandinsky6_mlx/utils/fixtures.py`: save/load a fixture as one safetensors file (`inputs.*`, `outputs.*`, `weights.*` prefixes) plus a JSON sidecar with config, seeds, library versions and the reference function name.
- `reference/tiny_configs.py`: tiny constructor arguments for every component, chosen so every code path runs:
  - DiT: `model_dim=32, model_dim_a=16, time_dim=16, ff_dim=64, ff_dim_a=32, num_text_blocks=1, num_visual_blocks=2, axes_dims=(2,2,4), axes_dims_a=(2,2,4), in_visual_dim=16, out_visual_dim=160, in_audio_dim=40, out_audio_dim=400, in_text_dim=8, in_text_dim2=4, visual_cond=True, ca_rope=True, cross_gates=True, fix_modulation=True, visual_token_type_num_embeddings=2`.
  - Video VAE: `block_out_channels=(8,8,8,8), layers_per_block=1, norm_num_groups=4` (same as the existing project validator).
  - Audio VAE and vocoder: smallest configs that keep the real layer structure; document the choice.
- `reference/make_tiny_fixtures.py`: builds each tiny model with `torch.manual_seed(0)`, float32, eval mode, runs it on NumPy-generated inputs, and saves weights, inputs and outputs. Also per-submodule fixtures for the DiT (each primitive and block) using forward hooks.
- `reference/record_real_activations.py`: given a checkpoint path, a component and a list of module names, runs the reference on given inputs and records the inputs/outputs of those modules with forward hooks. Written and unit-tested on a tiny model here; real runs happen in later specs (typically on Kaggle, then downloaded to `fixtures/real/`).
- `fixtures/tiny/*.safetensors` + `*.json` (committed; keep the total under 20 MB).
- `tests/unit/test_compare.py`, `tests/unit/test_fixtures.py`.
- `reports/01-reference-harness.json`.

## Implementation notes

- All random inputs come from `np.random.default_rng(seed)` (see `AGENTS.md`, Determinism).
- Fixture outputs for the DiT must include both the final output and every block output, so later specs can localize a mismatch to one block.
- For the scheduler, record `set_timesteps` results and every `step` output for a fixed sequence of model outputs.
- The recorder must store tensors in the dtype the reference computed them in and also in float32.

## Tests

- `compare` of identical arrays gives zeros and `cosine == 1`; of a known perturbation gives the expected `rel_l2`.
- Fixture round trip: save then load returns identical arrays and metadata.
- `make_tiny_fixtures.py` is deterministic: two runs produce byte-identical safetensors.
- The recorder captures the requested modules of a tiny DiT, and replaying a recorded block on its recorded input reproduces its recorded output in PyTorch (`rel_max <= 1e-6`).

## Validation runs

```bash
uv run --group reference python reference/make_tiny_fixtures.py --out fixtures/tiny
uv run --group reference python reference/make_tiny_fixtures.py --out /tmp/k6-tiny-2
cmp fixtures/tiny/dit.safetensors /tmp/k6-tiny-2/dit.safetensors
uv run --group reference pytest tests/unit -q
du -sh fixtures/tiny
```

## Acceptance criteria

- [ ] Tiny fixtures exist for: DiT (model and every primitive/block), video VAE decoder, audio VAE decoder, vocoder, PiFlow scheduler.
- [ ] Fixture generation is deterministic (byte-identical).
- [ ] `fixtures/tiny` total size is under 20 MB.
- [ ] All tests pass.

## Out of scope

MLX model code. Running the recorder on real weights.

## Stop conditions

A reference class cannot be constructed with a tiny config without editing diffusers → report `BLOCKED` with the error and the smallest config that works.
