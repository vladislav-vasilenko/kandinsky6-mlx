---
id: 02-weight-conversion
title: Convert diffusers safetensors into MLX weight files
depends_on: [01-reference-harness]
estimate: 1 day
real_weights: stage-b
---

## Goal

A converter that turns each diffusers component directory (`transformer/`, `vae/`, `audio_vae/`, `vocoder/`) into an MLX-ready safetensors file with the same parameter names, MLX tensor layouts and the project dtype policy. Later specs load models only through it.

## Read first

- MLX docs for `mlx.nn.Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`: weight layouts are channels-last (`[out, *kernel, in]` for convolutions); confirm the transposed-convolution layout in the installed MLX version.
- Header dtypes of the real checkpoint (`kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers@4c487e8b`):
  - `transformer`: 1976 BF16 tensors (5.90 GiB) and 423 F32 tensors (0.05 GiB).
  - `vae`, `audio_vae`, `vocoder`, `text_encoder_2`: all F32.
- `audio_vae/diffusion_pytorch_model.safetensors` also stores 667 `vocoder.*` tensors identical to `vocoder/` (see https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers/discussions/3). They must be dropped.

## Deliverables

- `src/kandinsky6_mlx/convert/convert.py` with `convert_component(src_dir, component, dst_dir, dtype_policy)` and a CLI `python -m kandinsky6_mlx.convert --checkpoint <dir> --out <dir>`.
- `src/kandinsky6_mlx/utils/dtype_policy.py`: rules for which parameters stay float32 (DiT: names containing `time_embeddings` or `modulation`, plus every tensor stored as F32 in the checkpoint), the default compute dtype for the rest (`bfloat16`), and a `float32` mode for parity tests.
- Output per component: `model.safetensors`, `config.json` copied unchanged, `conversion.json` with source revision, file SHA-256, tensor count, dropped keys and transposed keys.
- `tests/unit/test_convert.py`.
- `reports/02-weight-conversion.json`.

## Implementation notes

- Name mapping should be the identity: MLX modules will mirror diffusers module trees. Any rename needed later is added here, with a test.
- Transpose only convolution weights: Conv1d `[out,in,k] -> [out,k,in]`, Conv2d `[out,in,kh,kw] -> [out,kh,kw,in]`, Conv3d `[out,in,kd,kh,kw] -> [out,kd,kh,kw,in]`. Identify convolutions by module type in the reference model, not by tensor rank.
- `Linear` weights keep `[out, in]`.
- The video VAE `CausalConv3d` wraps an inner `conv`; the parameter path is `...conv.weight`.
- Never upcast BF16 tensors to save as F32 in default mode; keep the stored dtype unless the policy says float32.

## Tests

- Tiny fixtures: converting the tiny DiT, VAE, audio VAE and vocoder weights produces exactly the parameter names the MLX modules will expect (compare against `named_parameters` of the reference with conv layouts transposed).
- A Conv1d/Conv2d/Conv3d round trip: MLX convolution with converted weights equals the torch convolution on the same input (`rel_max <= 1e-5`, float32).
- `audio_vae` conversion drops every `vocoder.*` key and keeps all others.
- Dtype policy: tiny DiT `time_embeddings.*` and `*modulation*` weights are float32 after conversion in `bfloat16` mode.

## Validation runs

```bash
uv run --group reference pytest tests/unit/test_convert.py -q
# Stage B, only if the checkpoint is present locally:
uv run python -m kandinsky6_mlx.convert --checkpoint <path-to-snapshot> --out weights/lite-distill
```

Stage B report: tensor counts per component, dropped keys (expect 667 in `audio_vae`), output sizes, wall time.

## Acceptance criteria

- [ ] Stage A tests pass.
- [ ] Stage B (if run): every component converts; `audio_vae` loses exactly 667 tensors; converted DiT has 1976 bf16 + 423 f32 tensors or the policy-defined split, documented in the report.

## Out of scope

Quantization (spec 14). Text encoders (spec 11).

## Stop conditions

A tensor in the real checkpoint does not match any reference parameter name → `FAIL` with the list.
