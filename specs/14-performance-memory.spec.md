---
id: 14-performance-memory
title: Speed, memory and quantization study
depends_on: [09-pipeline-t2v-mvp, 10-video-vae-tiling]
estimate: 2-4 days
real_weights: required
---

## Goal

Make the port practical: measure, then reduce time and memory, without losing parity silently. Decide whether Pro is feasible on available Macs.

## Read first

- MLX docs: `mx.compile`, `mx.fast.*`, `nn.quantize`, memory APIs (`mx.get_peak_memory`, `mx.clear_cache`, `mx.set_wired_limit`).
- Reports of specs 09 and 10 (baseline numbers).

## Deliverables

- `scripts/benchmark.py`: per-stage wall time and peak memory for Smoke and Baseline presets, warm and cold.
- Optimizations behind flags: `mx.compile` of the DiT block, attention chunking for long sequences, VAE decode in bf16 (only if parity criterion holds), DiT weight quantization (8 and 4 bit, `Linear` only, keep fp32 modules).
- `reports/14-performance-memory.json` and `docs/PERFORMANCE.md` with a table per chip.

## Tests

- Every optimization keeps the spec 09 per-step cosine ≥ 0.999 vs the unoptimized MLX run, except quantization, which reports its drift and Smoke PSNR without a pass/fail threshold.
- Benchmark script runs in CI with tiny models (smoke test only).

## Validation runs

```bash
uv run python scripts/benchmark.py --weights weights/lite-distill --preset smoke --repeat 3
uv run python scripts/benchmark.py --weights weights/lite-distill --preset baseline --repeat 1
```

Compare with the Kaggle 2×T4 numbers recorded in the study repository (Smoke ≈ 13 s denoise, Baseline ≈ 721 s denoise, ≈ 694 s video decode).

## Acceptance criteria

- [ ] Benchmark table for at least one chip, Smoke and Baseline.
- [ ] Each optimization's speed/memory gain and parity effect recorded.
- [ ] Pro feasibility verdict: memory estimate per quantization level and, if a machine allows, one Pro-distill Smoke run.

## Out of scope

Custom Metal kernels (follow-up).

## Stop conditions

An optimization breaks parity → keep it off by default and record why.
