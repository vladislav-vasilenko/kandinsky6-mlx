---
id: 16-ane-dit
title: Experiment: DiT blocks on Core ML and the Neural Engine
depends_on: [05-dit-model, 14-performance-memory]
estimate: 1 day for the probe
real_weights: required
---

## Goal

Find out whether DiT blocks can run on the Neural Engine at all, and whether that is worth it. The DiT is the heaviest stage (about 50% of a Smoke generation, 78% at Baseline, 90% for Lite-5s), but also the hardest candidate. This is an experiment: a measured "no" is the expected and a valid result.

## What makes it hard

- **Precision.** diffusers keeps `time_embeddings` and `modulation` in float32 and computes `apply_scale_shift`, `apply_gate_sum`, RoPE and norm inputs in float32 inside every block (`transformer_kandinsky6.py`), and 423 tensors are stored as F32. ANE computes in 16-bit float. Reports 09 and 13 show that many steps amplify even bf16 rounding.
- **Mixed devices.** Keeping those operations off ANE means crossing devices many times per block.
- **Size.** About 6 GiB of weights, up to 50 220 video tokens at Baseline, a separate compiled model per token count.
- **This machine.** ANE may lose to a 40-core GPU here and still help on a base chip; a verdict needs a real low-end Mac for that second case.

## Read first

- `reports/05-dit-model.json`, `reports/14-performance-memory.json`.
- `src/kandinsky6_mlx/models/dit/`.

## Probe

Convert one fused block and one encoder block with the real weights, at Smoke token counts. Variants: everything in 16-bit float on ANE, and float32 operations left on the CPU or GPU.

Record per block:

- from `MLComputePlan`: which operations run on ANE, GPU and CPU;
- time and peak footprint vs MLX bf16 (the speed baseline: about 1.33 s per step at Smoke, from spec 14);
- `rel_l2` and cosine vs the float32 block, relative to the torch bf16 drift of the same block (`fixtures/real/dit_smoke`).

## Success criteria

To continue past the probe, a block must be at least as fast as MLX bf16 on some available chip, with drift within 2x torch bf16's (`AGENTS.md`), counting transfer costs between devices. Otherwise the verdict is no-go.

## Deliverables

- `reports/16-ane-dit.json` with the verdict and the numbers. Any further port is a new spec, after this verdict.

## Acceptance criteria

- [ ] Both block types and both variants measured.
- [ ] Go or no-go recorded with its reasons, and which machines it applies to.

## Out of scope

Porting the whole DiT; quantization and step skipping (spec 14).

## Stop conditions

Blocks fail to convert or fall back from ANE almost entirely → record it and stop.
