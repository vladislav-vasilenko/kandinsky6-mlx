---
id: 10-video-vae-tiling
title: Tiled and temporally tiled video VAE decoding
depends_on: [08-video-vae-decoder, 09-pipeline-t2v-mvp]
estimate: 2 days
real_weights: stage-b
---

## Goal

Decode Baseline-size latents (512×768, 121 frames) within the memory of a 32 GB Mac, with the same tiling and blending as diffusers so results match.

## Read first

- `autoencoder_kl_hunyuan_video.py`: `enable_tiling` (724), `blend_v/h/t` (851–873), `tiled_decode` (925), `_temporal_tiled_decode` (1014); defaults `tile_sample_min_height/width=256`, `tile_sample_min_num_frames=16`, strides `192/192/12`.
- Which tiling settings the study notebook uses for Baseline.

## Deliverables

- Tiling in `src/kandinsky6_mlx/models/video_vae/decoder.py` (`decode(z, tiling=TilingConfig|None)`).
- `tests/parity/test_video_vae_tiling.py`.
- Pipeline option `vae_tiling` and the Baseline preset.
- `reports/10-video-vae-tiling.json`.

## Implementation notes

- Match the reference tile order, overlap and linear blending exactly; tiling seams are where silent mismatches appear.
- Evaluate (`mx.eval`) each tile and release intermediates so peak memory scales with tile size, not video size.

## Tests

- Tiny model: tiled decode equals reference tiled decode (`rel_max <= 1e-4`) for spatial-only, temporal-only and both, with sizes that are not multiples of the stride.
- Real (stage B): Baseline latents from a recorded run → PSNR ≥ 40 dB vs reference tiled decode; peak memory recorded.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_video_vae_tiling.py -q
uv run python scripts/generate.py --preset baseline ... --out outputs/10/baseline.mp4
```

## Acceptance criteria

- [ ] Tiny tiling parity passes.
- [ ] Stage B: Baseline decode peak memory ≤ 20 GB and PSNR criterion met.

## Out of scope

Tiled encoding (spec 12 if needed).

## Stop conditions

Reference tiling itself produces visible seams → still match it; note it as an upstream observation.
