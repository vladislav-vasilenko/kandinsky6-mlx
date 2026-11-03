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

## Publishable benchmark (added after spec 13, owner's request)

The benchmark must answer, with numbers fit for the GitHub README and the Hugging Face card: is the MLX pipeline
faster than the reference diffusers pipeline on the same Mac, how much memory each uses, how cold and warm runs differ,
and what every mode and option costs.

Implementations:
- MLX: bf16 (default) and float32; each optimization of this spec on and off.
- Reference: diffusers 0.42.1 on Apple MPS as the study notebook runs it (DiT bf16, VAEs and vocoder float32,
  `vae.enable_tiling()` for Baseline). diffusers on CPU float32 for Smoke only, as context.

Matrix (each cell runs the same inputs: the same text embeddings, NumPy noise and image on both sides):
- Models: Lite-distill (PiFlow, 10 steps) and Lite-5s (FlowMatch, 50 steps, guidance 5; MLX sequential and batched CFG).
- Modes: text-to-video and image-to-video; with and without audio; VAE tiling on and off where it applies; text
  encoding (Qwen bf16 and 8-bit in MLX vs Qwen bf16 in torch on MPS, CLIP); the 16 GB profile (`mx.set_memory_limit`).
- Sizes: Smoke for the full matrix. Baseline: Lite-distill on MLX cold + 3 warm runs, diffusers on MPS one run after a
  short memory probe under the guard; Lite-5s with guidance one cold MLX run (about 2.2 h, overnight) and the diffusers
  time extrapolated from its per-step time, marked as an estimate.

Measurements per cell:
- Cold: the first call in a fresh process, including weight loading and kernel compilation.
- Warm: later calls in the same process, summarized per process.
- Statistics (owner's request; full method in BENCHMARK.md): the process is the unit of analysis; 10 independent
  processes for the 8 primary comparisons (an exact paired test cannot go below 2 / 2^n), 5 for other Smoke cells, 3 for
  Baseline, single labelled runs where a run takes hours; rounds in random seeded order with paired noise seeds; effect
  sizes with 95% intervals, exact sign-flip tests with the Holm correction; no outlier removal.
- Time per stage: text encoding, loading, denoising (and per step), video decode, audio decode, total.
- Memory, two ways so MLX and torch compare fairly: the framework's own peak (`mx.get_peak_memory`,
  `torch.mps.driver_allocated_memory`) and the process's physical footprint sampled every 0.2 s by the OS
  (`phys_footprint`); peak per stage and for the whole run.
- Machine facts: chip, GPU cores, unified memory, macOS, library versions, power source.

Outputs:
- `scripts/benchmark.py` (matrix selection by flags, one process per cold run, results appended as JSON lines) and the
  raw results in `benchmarks/results/<chip>/*.jsonl` plus a CSV summary.
- Charts (matplotlib, PNG and SVG) in `docs/benchmarks/`: time per stage MLX vs diffusers per mode; cold vs warm;
  peak memory per stage (framework and OS); Smoke vs Baseline; gain of each optimization; text encoding variants.
- `docs/PERFORMANCE.md` with the tables, charts, method and caveats; a short summary with a chart in the README; the
  main findings in a "Performance" section of the Hugging Face model card (an outward upload: owner's approval first).

Memory profiles 16, 24, 32 and 48 GB (owner's request). For each profile: the fastest settings that fit (about
60-70% of the RAM for the pipeline, leaving the rest to macOS and apps; 16 GB keeps the ~10 GB target above), the
resulting time and peak memory per stage, Smoke and Baseline. The settings that trade memory for speed are measured
one by one so the table shows what extra memory buys: VAE spatial tiling on/off and the conv chunk budget; clearing the
MLX cache between stages or keeping it (the first VAE decode in a process is ~2x slower); keeping the DiT and Qwen
resident across videos vs releasing them; batched (B=2) vs sequential CFG; Qwen bf16 vs 8-bit; DiT bf16 vs 8-bit.
Profiles are emulated on the development Mac with `mx.set_memory_limit` (and `torch.mps.set_per_process_memory_fraction`
for the reference): this shows what fits and what saving memory costs, not the speed of a real 16-32 GB Mac, whose
chip has fewer GPU cores and less bandwidth; the published tables say so, and runs on real machines are added when
available.

Rules: one heavy run at a time under the memory guard; every number states its cell and whether it is cold, warm or
an estimate; numbers measured earlier in reports 09-13 are not reused in the published tables (they were taken
without repeats and with recorder overhead), only cited as context.

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
- [ ] Publishable benchmark: the matrix above measured (cold and warm, time and memory per stage, MLX vs diffusers on
  MPS), raw results and charts committed, docs/PERFORMANCE.md and the README summary written, the HF card section
  prepared for the owner's approval.
- [ ] Memory profiles 16, 24, 32, 48 GB: fastest fitting settings, time and peak memory per stage for each, and the
  measured cost or gain of every memory-for-speed setting; emulation method and its limits stated.
- [ ] Each optimization's speed/memory gain and parity effect recorded.
- [ ] 16 GB profile: peak memory per stage under the limit at Smoke (and Baseline if reached), with the settings that achieve it and their parity or PSNR effect.
- [ ] Pro feasibility verdict: memory estimate per quantization level and, if a machine allows, one Pro-distill Smoke run.

## Out of scope

Custom Metal kernels (follow-up).

## Stop conditions

An optimization breaks parity → keep it off by default and record why.
