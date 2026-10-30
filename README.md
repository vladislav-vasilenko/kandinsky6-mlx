# kandinsky6-mlx

MLX port of [Kandinsky 6](https://github.com/kandinskylab/kandinsky-6) text/image-to-video+audio models for Apple Silicon.

**Status:** planning. Nothing is implemented yet; the work is split into stage specs in [`specs/`](specs/README.md).

## Scope

One code base for all six Diffusers checkpoints. They share the architecture and differ by config and sampling recipe:

| Checkpoint | DiT | Scheduler | Steps / guidance |
|---|---|---|---|
| Lite-distill-5s | 1792 dim, 32 blocks | PiFlow (`n_grid=10`) | 10 / 1.0 |
| Lite-5s, Lite-pretrain-5s | 1792 dim, 32 blocks | FlowMatchEulerDiscrete | 50 / 5.0 (CFG) |
| Pro-distill-5s | 4096 dim, 60 blocks | PiFlow (`n_grid=10`) | 10 / 1.0 |
| Pro-5s, Pro-pretrain-5s | 4096 dim, 60 blocks | FlowMatchEulerDiscrete | 50 / 5.0 (CFG) |

The video VAE (HunyuanVideo), audio VAE (MMAudio), vocoder (BigVGAN) and text encoders (Qwen2.5-VL-7B-Instruct, CLIP-L) are the same files in every checkpoint. Only `audio_vae/config.json` `scaling_factor` differs (0.417 or 0.5302).

Reference implementation: `diffusers==0.42.1`. Every MLX module is validated against it.

Hardware, disk and environment needs per phase: [`REQUIREMENTS.md`](REQUIREMENTS.md). Rules for executing specs: [`AGENTS.md`](AGENTS.md).

## Layout

```
src/kandinsky6_mlx/
  models/dit/          Kandinsky6Transformer3DModel
  models/video_vae/    AutoencoderKLHunyuanVideo (decoder, then encoder, tiling)
  models/audio_vae/    MMAudioVAE (decoder)
  models/vocoder/      MMAudioVocoder (BigVGAN)
  schedulers/          PiFlow, FlowMatchEulerDiscrete
  text/                Qwen2.5-VL hidden states, CLIP pooled embedding
  pipelines/           T2V / I2V pipeline, MP4 export
  convert/             diffusers safetensors -> MLX weights
  utils/               comparison metrics, dtype policy, IO
reference/             PyTorch reference runners that produce fixtures
fixtures/              tiny fixtures (committed) and real fixtures (not committed)
tests/unit|parity|e2e  pytest suites
reports/               one validation report per stage
specs/                 stage specs, one executor run each
```

## License

Code: MIT (to be confirmed). Kandinsky 6 weights: MIT, see the original model cards. Qwen2.5-VL and CLIP keep their own licenses.
