---
id: 14-performance-memory
title: Speed, memory and quantization study
depends_on: [09-pipeline-t2v-mvp, 10-video-vae-tiling]
estimate: 2-4 days
real_weights: required
---

## Goal

Make the port practical: measure, then reduce time and memory, without losing parity silently. Decide whether Pro is feasible on available Macs, and make Lite run on a 16 GB Apple Silicon laptop.

## 16 GB profile (added after spec 08)

A 16 GB Mac leaves about 10-11 GB for MLX once macOS and apps take theirs, so the profile targets a peak of about 10 GB for the whole Lite pipeline at Smoke (Baseline if it fits). Measured starting points (reports 05, 07, 08): DiT bf16 peak 8.05 GiB (Smoke) / 11.6 GiB (Baseline) including 2.2 GB of float32 modulation weights; VAE decode 20.3 GiB at Smoke with chunked convs; vocoder 3.4 / 4.75 GiB; Qwen 7B in bf16 16.6 GB.

- Load components one at a time and free each before the next (text encoder -> DiT -> VAE -> vocoder), with `mx.clear_cache()` between stages.
- Precompute the per-step modulation vectors for all steps (they depend only on the timestep and the pooled text) and free the float32 modulation weights: exact, about -2.2 GB (report 05).
- VAE: spatial tiling and a smaller conv chunk budget from spec 10, sized for the profile.
- Text: 4-bit Qwen or cached embeddings from spec 11.
- DiT: 8-bit `Linear` weights if the above is not enough (drift reported as below).
- Verify on the development Mac with `mx.set_memory_limit` at the profile budget; record whether a real 16 GB machine was available for a final run.

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
- [ ] 16 GB profile: peak memory per stage under the limit at Smoke (and Baseline if reached), with the settings that achieve it and their parity or PSNR effect.
- [ ] Pro feasibility verdict: memory estimate per quantization level and, if a machine allows, one Pro-distill Smoke run.

## Out of scope

Custom Metal kernels (follow-up).

## Stop conditions

An optimization breaks parity → keep it off by default and record why.
