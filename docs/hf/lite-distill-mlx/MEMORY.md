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
2. **Around 13–15 GiB, Torch MPS is about 10% faster at this size:** 28 s against 31 s. MLX's faster denoising is outweighed by its
   slower decoding once the cache is limited.
3. **Below that, only MLX has been measured.** It ran at 9.8 and 8.9 GiB, at the cost of time. Torch MPS's lowest reliable run was
   13.4 GiB, and it ran out of memory at every limit of 9.36 GiB and below. Limits between 9.36 and 11.2 GiB (watermark 0.26–0.29)
   have not been tried, so its exact floor is open.
4. **The earlier gap in loading and denoising came from the loading order,** not from the frameworks.
5. **The port now follows diffusers' float32 rule,** which lowers MLX's denoising peak by 1 GiB. The tables on this page were measured
   with the earlier rule; see below.

These are small samples (n = 3 to 10) on one Mac and one size, with no significance tests. Treat small differences as ties.

## Memory against time

| Implementation | Setting | Peak, GiB | Total, s | Video decode, s | Runs |
| --- | --- | ---: | ---: | ---: | ---: |
| Torch MPS | watermark 1.7 (default) | stopped above 43 | – | – | 1 |
| Torch MPS | watermark 1.0 | 38.9 | 28.1 | 11.7 | 9 |
| Torch MPS | watermark 0.5 | 20.5 | 27.6 | 11.3 | 3 |
| Torch MPS | watermark 0.35 | 14.9 | 27.7 | 11.4 | 3 |
| Torch MPS | watermark 0.3 | 13.0 in 1 run; out of memory in 2 | 28.2 | 11.9 | 3 |
| Torch MPS | watermark 0.3, media after denoising | 13.4 | 28.4 + 2 to load media | 12.0 | 3 (2 timed) |
| Torch MPS | watermark 0.25, with either loading order | out of memory | – | – | 6 |
| Torch MPS | watermark 0.2, media after denoising | out of memory | – | – | 3 |
| MLX | no cache limit (default) | 35.9 | 26.2 | 12.5 | 5 |
| MLX | cache 16 GiB | 22.7 | 30.4 | 16.8 | 5 |
| MLX | cache 8 GiB | 15.8 | 30.7 | 17.0 | 5 |
| MLX | cache 4 GiB | 13.0 | 31.2 | 17.5 | 5 |
| MLX | cache 1 GiB | 9.8 | 33.1 | 19.5 | 5 |
| MLX | cache 0 | 8.9 | 38.8 | 24.8 | 5 |

Limits are 9.36 GiB at watermark 0.25 and 11.23 GiB at 0.3. Video decoding asked for more than that: for example 8.38 GiB allocated
plus a 1.06 GiB request at 0.25. The default watermark run was stopped by the benchmark's memory guard, to protect the Mac from swap.
One timed run of watermark 0.3 with media after denoising is left out: a leftover process of ours was still using the GPU, and its
steps took 10 s instead of 1.6 s. Its memory reading is kept, as it is in line with the other two.

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

## The float32 rule: why the DiT took 1 GiB more in MLX

diffusers lists `_keep_in_fp32_modules = ["time_embeddings", "modulation"]`, but it compares each dot-separated part of a
parameter's name exactly (`models/model_loading_utils.py:238, :448`). So `visual_modulation`, `va_modulation` and the like do not match,
and only 4 tensors stay in float32. The port used to match by substring and keep 276. This is not a conversion error: the converted file
has the checkpoint's dtypes. Measured parameters:

| Variant | bf16, GiB | float32, GiB |
| --- | ---: | ---: |
| Torch MPS, bf16 | 5.914 | 0.010 |
| MLX, bf16, earlier rule | 4.894 | 2.052 |
| MLX, dtypes as stored | 5.943 in total | |

All of these are DiT tensors; the VAEs and the vocoder are F32 on both sides. Sizes are as stored in the checkpoint.

| Block of the original | Module | Tensors | In the file | Torch bf16 | MLX, earlier rule | Added by it |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `visual_transformer_blocks` (32), `video_dec_block` | `visual_modulation` | 64 | 505 MiB, BF16 | bf16 | float32 | +505 MiB |
| `visual_transformer_blocks` (32), `audio_dec_block` | `visual_modulation` | 64 | 252 MiB, BF16 | bf16 | float32 | +252 MiB |
| `visual_transformer_blocks` (32) | `va_modulation` | 64 | 140 MiB, BF16 | bf16 | float32 | +140 MiB |
| `visual_transformer_blocks` (32) | `av_modulation` | 64 | 112 MiB, BF16 | bf16 | float32 | +112 MiB |
| `video_text_transformer_blocks` (2) | `text_modulation` | 4 | 21 MiB, BF16 | bf16 | float32 | +21 MiB |
| `audio_text_transformer_blocks` (2) | `text_modulation` | 4 | 10 MiB, BF16 | bf16 | float32 | +10 MiB |
| `video_time_embeddings` | whole module | 4 | 4.5 MiB, F32 | bf16 | float32 | 0 |
| `audio_time_embeddings` | whole module | 4 | 2.8 MiB, F32 | bf16 | float32 | 0 |
| `out_layer` | `modulation` | 2 | 3.5 MiB, BF16 | float32 | float32 | same as Torch |
| `audio_out_layer` | `modulation` | 2 | 1.8 MiB, BF16 | float32 | float32 | same as Torch |

These layers compute each block's `scale`, `shift` and `gate` coefficients, and they are heavy because of their large outputs: 16128 × 512
weights per block in `video_dec_block`, which is part of the DiT, not of the VAE. Both sides apply the coefficients in float32.
Report: [05-dit-model.json](https://github.com/vladislav-vasilenko/kandinsky6-mlx/blob/main/reports/05-dit-model.json).

### Comparing the two rules

The same recorded inputs in MLX bf16, against the float32 diffusers reference. Frames PSNR and audio SNR in dB, higher is closer; a
fresh process per run. **The weights did not change and need no new download:** the rule applies when they load.

| Model, seed | Frames: earlier rule | Frames: diffusers' rule | Frames: Torch bf16 | Audio: earlier rule | Audio: diffusers' rule |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lite-distill, 42 | 27.37 | 42.65 | 27.71 | 16.29 | 7.58 |
| Lite-distill, 1 | 45.80 | 40.07 | 36.89 | 17.76 | 19.39 |
| Lite-distill, 2 | 32.97 | 37.11 | 31.82 | 17.09 | 21.24 |
| Lite-distill, 3 | 30.44 | 35.35 | 27.44 | 14.67 | 14.14 |
| Lite-5s, 42 | 28.17 | 24.00 | 23.87 | 2.90 | −0.77 |
| Lite-5s, 1 | 27.55 | 30.07 | 27.28 | 2.51 | 4.64 |

- **Memory:** denoising peak 8.12 → 7.10 GiB, by MLX's own counter.
- **Speed:** denoising time unchanged within these few runs.
- **Quality:** frames are closer on 4 seeds of 6 and further on 2, with a median change of +3.3 dB. The spread, from −5.7 to +15.3 dB, is mostly
  how bf16 rounding grows over the steps, so this shows **no clear loss, not an improvement**. Six pairs cannot rule out a small loss.
  Audio moves both ways too.
- **Agreed before the results:** rel_l2 within 2× Torch bf16's drift at every step (at most 1.37×), frames within 1 dB of Torch bf16 on each seed, a median
  loss of 2 dB at most, cosine above the tests' minimums, and a check by eye and ear. All were met, and the real-weight bf16 tests pass with unchanged thresholds.

### What the new rule changes in the sound

On Lite-5s, seed 42, nothing differs to the eye, but a sound at the end of the clip is much weaker. It is in the float32 reference, and **Torch bf16 loses it in the
same way**, so the port now does what the bf16 reference does:

| Variant | RMS in the last 0.2 s | Peak | Peak at | SNR to float32, dB |
| --- | ---: | ---: | ---: | ---: |
| Float32 reference | 0.0652 | 0.135 | 1.02 s | – |
| MLX, earlier rule | 0.0695 | 0.142 | 1.03 s | 2.91 |
| MLX, diffusers' rule | 0.0387 | 0.082 | 0.90 s | −0.80 |
| Torch MPS, bf16 | 0.0377 | 0.080 | 0.90 s | −0.36 |

This is one seed; on others the earlier rule is not closer. **If fidelity matters, especially for audio, use float32:** see the
[speed page](SPEED.md#float32-when-fidelity-matters).

## What the rest of the footprint is

From macOS's `footprint` tool, with everything loaded and after releasing the DiT. GiB, one run of each.

| Category | Torch, all loaded | Torch, DiT released | MLX, all loaded | MLX, DiT released |
| --- | ---: | ---: | ---: | ---: |
| GPU memory (`IOAccelerator`) | 9.26 | 5.03 | 8.74 | 1.67 |
| of which allocated by the framework | 8.44 | 2.52 | 8.60 | 1.66 |
| `Malloc Large` (CPU) | 1.49 | 1.49 | under 0.05 | under 0.05 |
| Other | 0.53 | 0.53 | 0.23 | 0.23 |

- **About 1.5 GiB of Torch's footprint is CPU memory that stays allocated.** It appears while loading the VAE and audio VAE, not the
  DiT. Asking the allocator to return free memory does not reduce it, and it survives releasing the DiT, so it is held, not just
  cached. The DiT goes through `from_pretrained`; those components go through this benchmark's own loader, so **part of Torch's
  footprint may come from our loader rather than from Torch.** What holds it has not been identified.
- **After the DiT is released, Torch's driver keeps about 2.5 GiB above what is allocated,** even after `torch.mps.empty_cache()`.
- **MLX returns its memory:** GPU memory falls to what is allocated. The reading right after clearing the cache was 0.6 GiB higher
  (footprint 3.19 GiB) and fell by the next one (1.90 GiB). Whether that is from time passing or from the call in between is not known.

## Not explained yet

- What holds the 1.5 GiB of `Malloc Large` in Torch, and whether a different loader removes it.
- Why Torch's driver keeps 2.5 GiB after `empty_cache()`.
- Why MLX's video decoding slows from about 12.5 s to about 17 s as soon as the cache is limited.

## Limits

- The tables above were measured with the earlier float32 rule. The cache-limit and memory-budget cells are being measured again with
  diffusers' rule, and the lowest MLX figures should fall.
- One Mac, one size. Nothing here was measured at 864×480 for Torch's watermarks, or on a Mac with less memory.
- Lower-watermark and load-order cells have 3 runs or fewer. Swap grew by up to 2.4 GB in the Torch cell with media components after
  denoising, and by up to 1.8 GB in the default cells of both implementations.

Method and commands: [BENCHMARK.md](https://github.com/vladislav-vasilenko/kandinsky6-mlx/blob/main/BENCHMARK.md).
