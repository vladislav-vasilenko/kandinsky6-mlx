# Stage specs

Each spec is one executor run. It states what to build, what to read first, which tests and validation runs prove the result, and what the report must contain. Rules shared by all specs are in [`AGENTS.md`](../AGENTS.md).

## Order and dependencies

```
00 bootstrap
└─ 01 reference harness
   ├─ 02 weight conversion
   │  ├─ 03 DiT primitives ─ 04 DiT blocks ─ 05 DiT model ─┐
   │  ├─ 07 audio decoder + vocoder ───────────────────────┤
   │  └─ 08 video VAE decoder ──────────────────────────────┤
   └─ 06 PiFlow scheduler ──────────────────────────────────┤
                                                            09 T2V pipeline (MVP)
                                                            ├─ 10 video VAE tiling
                                                            ├─ 11 text encoders
                                                            ├─ 12 I2V (needs 10)
                                                            ├─ 13 FlowMatch + CFG
                                                            └─ 14 performance and memory
                                                               15 packaging and release
```

| Spec | Result | Needs real weights | Estimate |
|---|---|---|---|
| [00](00-bootstrap.spec.md) | Project, tooling, CI on CPU | No | 0.5 d |
| [01](01-reference-harness.spec.md) | Comparison utils, tiny fixtures, reference runners | No | 1.5 d |
| [02](02-weight-conversion.spec.md) | Portable diffusers → MLX weight converter, Kaggle conversion kernel | Stage B (Kaggle) | 1.5 d |
| [03](03-dit-primitives.spec.md) | RoPE, modulation, attention, FFN, embeddings | No | 2 d |
| [04](04-dit-blocks.spec.md) | Text encoder block, fused video/audio block, output heads | No | 2 d |
| [05](05-dit-model.spec.md) | Full DiT, Lite and Pro configs | Stage B | 2 d |
| [06](06-piflow-scheduler.spec.md) | PiFlow scheduler | No | 1.5 d |
| [07](07-audio-decoder-vocoder.spec.md) | Audio VAE decoder and BigVGAN | Stage B | 2 d |
| [08](08-video-vae-decoder.spec.md) | HunyuanVideo causal 3D decoder | Stage B | 3 d |
| [09](09-pipeline-t2v-mvp.spec.md) | T2V pipeline from cached text embeddings | Stage B | 3 d |
| [10](10-video-vae-tiling.spec.md) | Tiled decode for Baseline size | Stage B | 2 d |
| [11](11-text-encoders.spec.md) | Qwen2.5-VL hidden states, CLIP pooled | Stage B | 2 d |
| [12](12-i2v.spec.md) | VAE encoder and reference-frame conditioning | Stage B | 2 d |
| [13](13-flowmatch-cfg.spec.md) | Non-distilled checkpoints: FlowMatch Euler and CFG | Stage B | 2 d |
| [14](14-performance-memory.spec.md) | `mx.compile`, memory, quantization study | Yes | 2–4 d |
| [15](15-packaging-release.spec.md) | CLI, model cards, weight upload | Yes | 1–2 d |

MVP (first video with sound on a Mac): specs 00–09, about 15–20 working days.

## Stage A and stage B

Most specs have two stages:

- **Stage A** uses tiny random-weight models built from the reference classes. It runs on any Mac with MLX, needs no downloads, and must always pass.
- **Stage B** uses real checkpoint weights and reference activations produced from them. It runs only when the weights are present locally (see `AGENTS.md`, Environment). If they are not, the spec ends as `PARTIAL`.

## Writing a new spec

Copy [`_template.spec.md`](_template.spec.md). Every acceptance criterion must name a command or a metric that can be checked by someone else.
