# Speed benchmark: MLX vs Torch MPS

Apple M3 Max, 40-core GPU, 48 GB, macOS 27.0.1, AC power. MLX 0.32.4, PyTorch 2.15.0, diffusers 0.42.1, bf16. Back to the
[model card](README.md). Method and commands: [BENCHMARK.md](https://github.com/vladislav-vasilenko/kandinsky6-mlx/blob/main/BENCHMARK.md).

Text encoding is separate and not included. Totals leave out loading unless stated, because the two sides load differently: see below.
Everything here was measured with the port's earlier float32 rule. Denoising time did not change with the current one in a few runs:
see the [memory page](MEMORY.md).

## Main findings

1. **With 50 steps (Lite-5s) MLX takes about 10–12% less time** at 432×256. Tested over 9 pairs.
2. **With 10 steps (Lite-distill) there is no demonstrated difference** at 432×256. MLX's faster denoising and Torch's faster decoding cancel out.
3. **At 864×480 with 10 steps, MLX took 935 s against 1024 s.** Three clean runs each, all in agreement, but too few for the test.
4. **A Torch MPS denoising step takes 17–20% longer,** in every configuration at 432×256 and 11% longer at 864×480.
5. **A 5 s clip at 864×480 takes about 35 times as long as a 1 s clip at 432×256.** See below.

## How long a clip takes

Cold, median, without loading.

| Clip | Model and steps | MLX, s | Torch MPS, s | Runs |
| --- | --- | ---: | ---: | ---: |
| 1 s, 432×256, 25 frames | Lite-distill, 10 | 26.6 | 28.1 | 10 / 9 |
| 1 s, 432×256, 25 frames | Lite-5s, 50 + CFG | 147.0 | 168.9 | 10 / 9 |
| 5 s, 864×480, 121 frames | Lite-distill, 10 | 935 | 1024 | 3 / 3 |
| 5 s, 864×480, 121 frames | Lite-5s, 50 + CFG | 7592 (2 h 7 min) | not measured | 1 / – |

### Why 5 s at 864×480 takes so much longer

MLX, Lite-distill, going from 432×256 and 25 frames to 864×480 and 121 frames:

| Quantity | 432×256 | 864×480 | Growth |
| --- | ---: | ---: | ---: |
| Video tokens | 3 024 | 50 220 | 16.6× |
| Seconds per denoising step | 1.34 | 72.0 | 54× |
| Video decoding, s | 12.5 | 215 | 17× |
| Total, s | 26.6 | 935 | 35× |

Denoising grows about three times faster than the token count, and at this size it is about 77% of the time. Why it grows that much has not been
broken down yet; attention over more tokens is the obvious suspect, not a measured cause.

At 864×480, cold, 3 clean benchmark processes each:

| Implementation | Denoise, s | Seconds per step | Video decode, s | Audio, s | Total, s | Peak memory, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLX | 718 | 72.0 | 215 | 1.7 | 935 | 36.5 |
| Torch MPS | 798 | 80.0 | 225 | 1.3 | 1024 | 38.5 |

## Float32 when fidelity matters

In bf16, rounding grows over the steps and can change details and sounds, on Torch MPS too. `--dtype float32` matches the float32 reference (frames at 96 dB or more),
and costs little time at this size. MLX, 432×256, cold, without loading, with the default cache:

| Model and steps | bf16, s | float32, s | Extra time | Denoising memory, bf16, GiB | Denoising memory, float32, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lite-distill, 10 | 26.6 | 27.5 | 3% | 8.7 | 13.7 |
| Lite-5s, 50 + CFG | 147.0 | 166.4 | 13% | 8.7 | 13.6 |

Float32 has 3–4 runs, against 10 for bf16, and was not tested for significance; swap grew in some of its rounds. Bf16 here uses the earlier float32 rule, and the current one is about 1 GiB lower. Both peak at about 35.9 GiB overall, which is the cache's ceiling. Float32 with a cache limit or at 864×480 has not been measured, and it will not fit the lowest memory settings.

## Each published clip

One cold run per clip with seed 42, in seconds, with loading **included in the total**. These are examples, not statistics.

{{CLIP_TIMES}}

**Read the 5 s rows with care.** They ran while lighter work (tests, encoding, uploads) was also on the Mac, and they are slower than the clean benchmark
above: MLX's video decoding took 269–276 s against 215 s, and its denoising 736–762 s against 718 s. This narrows or removes MLX's lead in these rows, so use the benchmark
for comparing speed.

## Where the time goes at 432×256


Median seconds per stage, warm, text-to-video.

| Model | Implementation | Denoise | Seconds per step | Video decode | Audio | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Lite-distill | MLX | 13.4 | 1.34 | 13.6 | 0.4 | 10 |
| Lite-distill | Torch MPS | 15.9 | 1.57 | 10.5 | 0.2 | 9 |
| Lite-5s | MLX | 133.3 | 2.67 | 12.8 | 0.4 | 10 |
| Lite-5s | Torch MPS | 156.1 | 3.12 | 12.4 | 0.2 | 9 |

- **Denoising is where MLX gains:** a Torch MPS step takes 17–20% longer. With 50 steps this decides the total; with 10 steps
  Torch MPS's faster and steadier video decoding cancels it.
- **MLX's video decoding is uneven at 432×256:** usually about 12.7 s, sometimes 20–35 s. The cause is not established yet.
- **Loading:** about 1.3 s in MLX against 10–12 s in Torch MPS, but not like for like: MLX defers its decoders' weights to
  decoding. It is left out of the totals above.
- **Caveats:** this is one Mac. Swap grew in many rounds on both sides, and too few rounds were free of it (2–6) to test them
  on their own. An outside event filled the disk during the run; the 14 rounds it stopped are being repeated.
