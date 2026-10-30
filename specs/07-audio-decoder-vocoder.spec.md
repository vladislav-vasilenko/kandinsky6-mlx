---
id: 07-audio-decoder-vocoder
title: MMAudio VAE decoder and BigVGAN v2 vocoder
depends_on: [02-weight-conversion]
estimate: 2 days
real_weights: stage-b
---

## Goal

Audio latents `[B, 20, N]` → mel spectrogram → 44.1 kHz waveform in MLX, matching diffusers.

## Read first

- `diffusers/models/autoencoders/autoencoder_mmaudio.py`: `normalize`, `mp_silu`, `mp_sum` (~78–103), `MMAudioMPConv1d` (105), `MMAudioResnetBlock1D` (116), `MMAudioAttnProcessor`/`MMAudioAttnBlock1D` (135–168), `MMAudioUpsample1D` (170), `MMAudioDecoder1D` (240–289, note `ACTIVATION_CLIP` clamps and `learnable_gain`), `MMAudioVAE.decode` (421).
- `diffusers/models/autoencoders/mmaudio_vocoder.py`: `MMAudioSnakeBeta` (55), `MMAudioLowPassFilter1d` (69), `MMAudioUpSample1d` (84), `MMAudioDownSample1d` (103), `MMAudioActivation1d` (112), `MMAudioAMPBlock` (125), `MMAudioVocoder` (154).
- Released vocoder config: `num_mels=128`, `upsample_rates=[8,4,2,2,2,2]`, `upsample_kernel_sizes=[16,8,4,4,4,4]`, `upsample_initial_channel=1536`, `resblock_kernel_sizes=[3,7,11]`, dilations `[1,3,5]`.
- How the pipeline scales audio latents before decode (`scaling_factor` 0.417 or 0.5302 from the checkpoint's `audio_vae/config.json`).

## Deliverables

- `src/kandinsky6_mlx/models/audio_vae/decoder.py` (`MMAudioVAE` with `decode` only), `src/kandinsky6_mlx/models/vocoder/bigvgan.py`.
- `tests/parity/test_audio_decoder.py`, `tests/parity/test_vocoder.py`, stage-B `tests/parity/test_audio_real.py`.
- `reports/07-audio-decoder-vocoder.json`.

## Implementation notes

- `normalize` uses float32 vector norm and the `sqrt(norm.numel()/x.numel())` factor; the MP conv multiplies weights by `gain` per call.
- Weight norm is already folded in the released weights (no `weight_g`/`weight_v`); confirm by key names.
- BigVGAN anti-aliased activation: upsample ×2 with a Kaiser-windowed sinc filter (transposed conv with fixed filter and replicate padding), SnakeBeta (`x + 1/(beta+eps) * sin(alpha*x)^2`, alpha/beta stored in log scale — check), downsample ×2. Padding details decide parity; test the filter blocks alone first.
- `ConvTranspose1d` for the upsampling stages: verify MLX output length equals torch (`(L-1)*stride - 2*pad + kernel`).

## Tests

- Stage A: each primitive (MP conv, resnet block, attention block, upsample; Snake, filters, AMP block) and the whole tiny decoder/vocoder in float32 (`rel_max <= 1e-4`).
- Output lengths: for latent length 45 (Smoke) and the Baseline length, the waveform length equals the reference.
- Stage B: real weights, recorded latents from a Smoke run → mel and waveform; waveform `rel_l2 <= 1e-3` in float32, plus SNR in dB in the report.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_audio_decoder.py tests/parity/test_vocoder.py -q
uv run pytest tests/parity/test_audio_real.py -m real_weights -q
```

Write `outputs/07/mlx.wav` and `outputs/07/reference.wav` for listening.

## Acceptance criteria

- [ ] Stage A parity passes.
- [ ] Stage B (if run): waveform criterion met; both WAV files written and their durations equal.

## Out of scope

Audio VAE encoder (not needed for T2V/I2V).

## Stop conditions

MLX lacks a needed op (e.g. a padding mode) → implement it explicitly and test it; if impossible, `BLOCKED` with the op name.
