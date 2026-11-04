# Memory benchmark: MLX vs Torch MPS

Kandinsky 6.0 Lite-distill, 432×256, 25 frames, 10 steps, text-to-video with audio, bf16. Apple M3 Max, 40-core GPU, 48 GB, macOS
27.0.1, MLX 0.32.4, PyTorch 2.15.0, diffusers 0.42.1. Back to the [model card](README.md).

**Memory** is the process's peak physical footprint (`ri_phys_footprint`) sampled from outside the process, GPU buffers included.
**Time** is the median cold total without loading, in seconds. Every run is an independent process. Text encoding is separate and
not included.

## Main findings

1. **Both default peaks are mostly cache.** Torch MPS at `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` peaks at 38.9 GiB; at 0.35 the same
   run peaks at 14.9 GiB in the same time. So its 38.9 GiB says nothing about what it needs. MLX's cache does buy speed in video
   decoding, so limiting it costs time.
2. **Around 15 GiB, Torch MPS is about 10% faster at this size:** 27.7 s against 30.7 s. MLX's faster denoising is outweighed by its
   slower decoding once the cache is limited.
3. **MLX has been measured lower, down to 8.9 GiB.** Torch MPS's minimum is not established: it failed with the 0.25 limit and
   nothing between 0.25 and 0.35 has been tried. Torch's denoising alone can be smaller than MLX's (see below), so no
   minimum-memory winner is claimed.
4. **The earlier gap in loading and denoising came from the loading order,** not from the frameworks.

These are small samples (n = 3 to 10) on one Mac and one size, with no significance tests. Treat small differences as ties.

## Memory against time

| Implementation | Setting | Peak, GiB | Total, s | Video decode, s | Runs |
| --- | --- | ---: | ---: | ---: | ---: |
| Torch MPS | watermark 1.7 (default) | stopped above 43 | – | – | 1 |
| Torch MPS | watermark 1.0 | 38.9 | 28.1 | 11.7 | 9 |
| Torch MPS | watermark 0.5 | 20.5 | 27.6 | 11.3 | 3 |
| Torch MPS | watermark 0.35 | 14.9 | 27.7 | 11.4 | 3 |
| Torch MPS | watermark 0.25 | out of memory | – | – | 3 |
| MLX | no cache limit (default) | 35.9 | 26.2 | 12.5 | 5 |
| MLX | cache 16 GiB | 22.7 | 30.4 | 16.8 | 5 |
| MLX | cache 8 GiB | 15.8 | 30.7 | 17.0 | 5 |
| MLX | cache 4 GiB | 13.0 | 31.2 | 17.5 | 5 |
| MLX | cache 1 GiB | 9.8 | 33.1 | 19.5 | 5 |
| MLX | cache 0 | 8.9 | 38.8 | 24.8 | 5 |

At watermark 0.25 the limit was 9.36 GiB; with 8.18 GiB allocated, a 1.22 GiB allocation failed. That run had all its components
loaded from the start, so this is not necessarily Torch MPS's floor. The default watermark run was stopped by the benchmark's
memory guard, to protect the Mac from swap.

## The weight-loading order

In the main benchmark, Torch holds the video VAE, audio VAE and vocoder from the start, encoders included. MLX loads only their
decoders, lazily, when decoding starts. To compare like for like, each side also ran the other's order. Medians over 3 runs, except
for the first row of each pair (Torch 9, MLX 10).

| Implementation | Media components | Memory in denoising, GiB | Denoising, s |
| --- | --- | ---: | ---: |
| Torch MPS | all at the start | 11.2 | 16.2 |
| Torch MPS | after denoising | 7.0 | 16.2 |
| MLX | lazily, decoders only (default) | 8.7 | 13.4 |
| MLX | all at the start, decoders only | 10.4 | 13.4 |

- With the same order the difference is 1–2 GiB, and it is **in Torch's favour** when media components load later.
- **Denoising time does not depend on the order,** so MLX's faster steps are real.
- Loading media components after denoising does not reduce Torch's overall peak at watermark 1.0 (39.3 GiB), which is set in video
  decoding, by the cache.

## Step by step: what each load adds

Physical footprint in GiB after each step, one fresh process per framework, one run of each. No generation runs here.

| Step | MLX | | Torch MPS |
| --- | ---: | --- | ---: |
| Libraries imported | 0.03 | | 0.33 |
| DiT | 7.33 | | 6.77 |
| + video VAE | 7.87 | decoder only / with encoder | 8.68 |
| + audio VAE | 8.53 | decoder only / with encoder | 10.35 |
| + vocoder | 8.95 | | 11.31 |
| DiT released, cache cleared | 2.94 | | 7.09 |

With everything loaded, the gap of 2.36 GiB adds up as follows:

| Part | GiB |
| --- | ---: |
| Weights (MLX 8.60, Torch 8.44): Torch's encoders add 0.86, but MLX's DiT takes 1.02 more | −0.16 |
| MPS driver memory above Torch's allocations | +0.82 |
| The rest of the footprint: Torch 2.05, MLX 0.35 | +1.70 |

## Not explained yet

- Why the DiT takes 6.95 GiB in MLX when its file is 5.94 GiB. Torch allocates 5.93 GiB.
- What the 2 GiB of Torch's footprint outside the MPS driver is. Leftover CPU copies from loading are a guess, not a measurement.
- Why Torch stays at 7.09 GiB after releasing the DiT, with 2.52 GiB allocated.
- Why MLX's video decoding slows from about 12.5 s to about 17 s as soon as the cache is limited.

## Limits

- One Mac, one size. Nothing here was measured at 864×480 for Torch's watermarks, or on a Mac with less memory.
- Lower-watermark and load-order cells have 3 runs. Swap grew by up to 2.4 GB in the Torch cell with media components after
  denoising, and by up to 1.8 GB in the default cells of both implementations.

Method and commands: [BENCHMARK.md](https://github.com/vladislav-vasilenko/kandinsky6-mlx/blob/main/BENCHMARK.md).
