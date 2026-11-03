---
id: 15-ane-vae
title: "Experiment: the video VAE decoder on Core ML and the Neural Engine"
depends_on: [08-video-vae-decoder, 10-video-vae-tiling, 14-performance-memory]
estimate: 1 day for the probe; 3-5 days if it passes
real_weights: required
---

## Goal

Find out whether the video VAE decoder can run on Core ML and the Neural Engine with a much lower peak memory than MLX, and port it if so. This is an experiment: a measured "no" is a valid result.

**Primary success metric: peak physical footprint of the video decode stage** (`ri_phys_footprint`, as in `scripts/benchmark.py`), at the same size and with the same tiling as the MLX numbers it is compared with.

Guards: frames PSNR >= 40 dB vs the float32 reference; decode time not worse than the MLX default (median from the spec 14 `tradeoffs` cells).

Reference points from spec 14: about 36 GiB and 12.7 s by default, about 6.5 GiB and 25 s with MLX's cache limit 0 (replace them with the final benchmark's medians). To be worth keeping, the port must use less memory than the second, or match its memory at the first's speed.

## Read first

- `reports/08-video-vae-decoder.json`, `reports/10-video-vae-tiling.json`, `reports/14-performance-memory.json`.
- `src/kandinsky6_mlx/models/video_vae/decoder.py`.
- coremltools documentation: conversion, `MLComputePlan`, compute units, precision.

## Stage 1: probe (do this first)

Convert one resnet block of the last up block (128 channels, the largest activations) with the real weights. ANE's rank limit means 5D tensors must be flattened: a causal 3x3x3 conv decomposes exactly into three 2D convs over its kernel slices, with time in the batch dimension, `y[t] = conv2d(x[t], W0) + conv2d(x[t-1], W1) + conv2d(x[t-2], W2)`.

Record, for a 17-frame 256x256 tile:

- from `MLComputePlan`: which operations run on ANE, GPU and CPU;
- time and peak footprint on ANE vs MLX;
- output difference vs float32 (ANE computes in 16-bit float).

Open questions to answer: GroupNorm statistics over all frames of a tile, the mid block's causal attention mask, fixed input shapes for the smaller edge tiles (reference tile sizes must not change, because tile size changes the output).

## Stage 2: port (only if stage 1 can plausibly meet the metric)

- `src/kandinsky6_mlx/models/video_vae/coreml.py` and a conversion script.
- Pipeline option to use it, off by default.
- Parity test against the reference tiled decode; benchmark cells at Smoke and Baseline.

## Deliverables

- `reports/15-ane-vae.json` with the verdict and the numbers, whichever way it goes.

## Acceptance criteria

- [ ] Stage 1 measured; go or no-go recorded with its reasons.
- [ ] If go: the primary metric and both guards measured at Smoke and Baseline.

## Out of scope

The DiT (spec 16). Other Core ML runtimes.

## Stop conditions

Most of the block's operations fall back from ANE, or the block's memory or time on ANE is not better than MLX → stop, record the numbers, and keep MLX.
