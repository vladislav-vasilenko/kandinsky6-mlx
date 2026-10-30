---
id: 08-video-vae-decoder
title: HunyuanVideo causal 3D VAE decoder (no tiling)
depends_on: [02-weight-conversion]
estimate: 3 days
real_weights: stage-b
---

## Goal

Video latents `[B, 16, T, H/8, W/8]` → RGB frames `[B, 3, 4(T-1)+1, H, W]` in MLX, matching diffusers at Smoke size without tiling.

## Read first

`diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py` (0.42.1):

- `HunyuanVideoCausalConv3d` (46–77): replicate padding on time only at the front (`time_causal_padding`), symmetric spatial padding.
- `HunyuanVideoUpsampleCausal3D` (79–119): first frame upsampled spatially only, other frames ×2 in time and space, `mode="nearest"`.
- `HunyuanVideoResnetBlockCausal3D` (141), `HunyuanVideoMidBlock3D` (187, attention with a causal frame mask), `HunyuanVideoUpBlock3D` (347), `HunyuanVideoDecoder3D` (518).
- `AutoencoderKLHunyuanVideo._decode`/`decode` (805–849); framewise decoding flags (707–722).
- Config: `block_out_channels=[128,256,512,512]`, `layers_per_block=2`, `norm_num_groups=32`, `scaling_factor=0.476986`, temporal ×4, spatial ×8.

## Deliverables

- `src/kandinsky6_mlx/models/video_vae/decoder.py`.
- `tests/parity/test_video_vae_decoder.py`, stage-B `tests/parity/test_video_vae_real.py`.
- `reports/08-video-vae-decoder.json`.

## Implementation notes

- MLX convolutions are channels-last: keep tensors `(B, T, H, W, C)` inside the module and convert at the boundary; the converter (spec 02) already transposed weights.
- GroupNorm: diffusers normalizes over `(C/groups, T, H, W)`; make sure MLX `GroupNorm` with channels-last matches (use `pytorch_compatible=True` if available, otherwise implement).
- The mid-block attention uses a causal mask over frames (`prepare_causal_attention_mask`); reproduce it.
- Decoding Smoke latents (`[1,16,7,32,54]` → 25 frames 256×432) fits in memory without tiling; Baseline needs spec 10.
- Compute in float32 (weights are F32); a bf16 mode is studied in spec 14.

## Tests

- Stage A: causal conv (padding semantics on a distinct-integer tensor), upsample (frame count `1 + 2(T-1)` per stage), resnet, mid-block attention mask, tiny decoder end to end (`rel_max <= 1e-4`).
- Stage B: real weights, recorded Smoke latents → frames; `rel_l2 <= 1e-3` in float32, PSNR in dB against reference frames in the report.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_video_vae_decoder.py -q
uv run pytest tests/parity/test_video_vae_real.py -m real_weights -q
```

Save `outputs/08/mlx.mp4` and `outputs/08/reference.mp4` plus a side-by-side PNG of frames 0, 12, 24.

## Acceptance criteria

- [ ] Stage A parity passes; frame counts match for T ∈ {1, 2, 7}.
- [ ] Stage B (if run): criterion met, PSNR ≥ 40 dB, peak memory recorded.

## Out of scope

Tiling (spec 10), encoder (spec 12).

## Stop conditions

Peak memory at Smoke size exceeds 24 GB → record and stop; tiling spec handles it.
