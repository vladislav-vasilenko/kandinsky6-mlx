---
license: mit
library_name: mlx
pipeline_tag: text-to-video
base_model: kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers
tags:
- mlx
- apple-silicon
- video-generation
- text-to-video
- image-to-video
- audio-generation
- kandinsky
---

# Kandinsky 6.0 Lite-distill 5s — MLX

Video with synchronized audio on Apple Silicon: weights of
[kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers)
(revision `4c487e8b`) converted for [MLX](https://github.com/ml-explore/mlx), with a native pipeline:
[github.com/vladislav-vasilenko/kandinsky6-mlx](https://github.com/vladislav-vasilenko/kandinsky6-mlx).

This is an independent community port, not an official release. All credit for the model goes to
[Kandinsky Lab](https://kandinskylab.ai/models/video/); see the [technical report](https://arxiv.org/abs/2610.05608).

**Apple M3 Max, 48 GB, compared with the standard weights in diffusers on Torch MPS: with 50 guided steps (Lite-5s) MLX takes
about 10–12% less time, a tested difference. With the 10 steps of this distilled model there is no demonstrated difference at
432×256 (about 27 s for a 1 s clip with sound, without loading). A 5 s clip at 864×480 took 935 s against 1024 s (3 runs, not
tested).** See [Speed and memory](#speed-and-memory).

Works: text-to-video and image-to-video with audio, prompt to MP4 in one command, 432×256 and 864×480.
All components are compared with diffusers on real weights: see [Verification](#verification).

## Video examples: Torch MPS vs MLX

**Standard weights in diffusers on Torch MPS vs this repository in MLX · same prompts, text embeddings, noise and seed ·
432×256 · 25 frames · 24 FPS · 10 PiFlow steps · bf16 DiT · audio.**

The clips show the same scene and differ in details: both sides run the DiT in bf16, and over 10 distilled steps its
rounding differences grow differently in each (peak signal-to-noise between the pairs: 23.5–40.7 dB). Against a float32
reference, MLX bf16 drifts no more than Torch bf16: see [Verification](#verification).

### Fox in the snow

{{PAIR_FOX}}

{{PROMPT_FOX}}

### Fashion film

{{PAIR_WOMAN}}

{{PROMPT_WOMAN}}

### Retrofuturist tram

{{PAIR_TRAM}}

{{PROMPT_TRAM}}

### Biomechanical flower

{{PAIR_BLOOM}}

{{PROMPT_BLOOM}}

### Scandinavian room

{{PAIR_ROOM}}

{{PROMPT_ROOM}}

### Image to video

The video starts from the input image, with the Fashion film and Retrofuturist tram prompts.

{{PAIR_I2V_WOMAN}}

{{PAIR_I2V_TRAM}}

### Frame comparison

Frame 12 (0.5 s) of the text-to-video and image-to-video clips above. Left: Torch MPS. Right: MLX. Click for full size.

<a href="{{FRAME_SHEET}}"><img src="{{FRAME_SHEET}}" alt="Frame 12 of each clip: Torch MPS and MLX" width="100%"></a>

### 5 seconds at 864×480

**MLX only · 864×480 · 121 frames · 24 FPS · 10 steps · audio · the Fashion film preset's long prompt.** A Torch MPS pair
at this size will come with the full benchmark.

{{BASELINE_VIDEO}}

## Speed and memory

**Apple M3 Max, 40-core GPU, 48 GB, macOS 27.0.1, AC power · MLX 0.32.4 · PyTorch 2.15.0, diffusers 0.42.1.**
Both sides use the same inputs, VAE tiling on and the DiT freed before decoding.
Torch MPS runs with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`; at its default of 1.7 it passed 43 GiB on this 48 GB Mac
and the benchmark's memory guard stopped it. The speed tables use 1.0, whose peak is mostly cache: see [Memory](#memory).

> **Status.** Speed at 432×256 is final apart from a few repeated rounds; it comes from independent processes in a random
> order, paired by seed. Still to come: Torch MPS's minimum memory. Method and commands: [BENCHMARK.md](https://github.com/vladislav-vasilenko/kandinsky6-mlx/blob/main/BENCHMARK.md).

### Time: Torch MPS vs MLX

Total time without loading, in seconds: 432×256, 25 frames, 9 paired processes. **Cold** is the first generation in a fresh process;
**warm** is a later one in the same process.

| Model and mode | Run | Torch MPS, s | MLX, s | Torch time / MLX time (95% CI) | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Lite-5s, 50 steps + CFG, text-to-video | cold | 168.9 | 147.0 | 1.14 (1.12–1.15) | **MLX about 12% faster** |
| | warm | 168.8 | 146.5 | 1.12 (1.08–1.16) | **MLX about 11% faster** |
| Lite-5s, 50 steps + CFG, image-to-video | cold | 192.2 | 167.2 | 1.14 (1.10–1.17) | **MLX about 12% faster** |
| | warm | 193.0 | 174.4 | 1.11 (1.06–1.16) | **MLX about 10% faster** |
| Lite-distill, 10 steps, text-to-video | cold | 28.1 | 26.6 | 1.01 (0.92–1.10) | no demonstrated difference |
| | warm | 26.6 | 27.4 | 0.96 (0.88–1.04) | no demonstrated difference |
| Lite-distill, 10 steps, image-to-video | cold | 30.0 | 28.7 | 1.00 (0.93–1.08) | no demonstrated difference |
| | warm | 29.8 | 29.0 | 0.96 (0.87–1.05) | no demonstrated difference |

Seconds are medians. The ratio above 1 means MLX is faster; it is the geometric mean of the paired runs, so unlike the medians it
includes MLX's occasional slow video decodes (below). Each ratio is tested with an exact sign-flip test, Holm-corrected over the 8
tests: p = 0.031 for the four MLX results, and p above 0.05 for the others. A claim needs p < 0.05, an interval excluding 1, and at
least a 5% difference.

### Where the time goes

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

### Memory

**Full tables, causes and open questions: [memory benchmark](https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/blob/main/MEMORY.md).**
Peak physical footprint at 432×256; total time without loading.

| Implementation | Setting | Peak, GiB | Total, s | Runs |
| --- | --- | ---: | ---: | ---: |
| Torch MPS | watermark 1.0 | 38.9 | 28.1 | 9 |
| Torch MPS | watermark 0.35 | 14.9 | 27.7 | 3 |
| Torch MPS | watermark 0.25 | out of memory | – | 3 |
| MLX | default | 35.9 | 26.2 | 5 |
| MLX | cache limit 8 GiB | 15.8 | 30.7 | 5 |
| MLX | cache limit 1 GiB | 9.8 | 33.1 | 5 |
| MLX | cache limit 0 | 8.9 | 38.8 | 5 |

- **Default peaks are mostly cache, on both sides.** For Torch MPS, lowering the watermark costs almost no time; set
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO` lower than our 1.0 if you run it.
- **Around 15 GiB Torch MPS is about 10% faster at this size.** MLX has been measured lower, down to 8.9 GiB, but Torch MPS's
  minimum is not established, so no minimum-memory winner is claimed.
- **The gap in loading and denoising came from the loading order,** not the frameworks: with the same order the difference is
  1–2 GiB, and in Torch's favour when media components load after denoising.
- In MLX, limit the cache with `--cache-limit-gib`.

### Settings for less memory

Cache and memory limits meant for smaller Macs. **They ran on this M3 Max, so the times do not predict a smaller chip, and no
smaller Mac has been tested.** Prompt encoding is not included: see below.

| Memory of the target Mac | Cache limit, GiB | Memory limit, GiB | 432×256, s (5 runs) | Memory, GiB | 864×480, s (1 run) | Memory, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 GB | 16 | 32 | 30.3 | 23.1 | 1023 | 24.9 |
| 32 GB | 8 | 22 | 30.6 | 15.8 | 1051 | 16.5 |
| 24 GB | 4 | 16 | 31.1 | 11.7 | 1083 | 12.8 |
| 16 GB | 1 | 10 | 32.9 | 8.7 | 1111 | 11.0 |

### Prompt encoding

Warm encode of the 238-token Smoke prompt, median over independent processes, with the text encoders loaded (about 3–10 s
cold). The encoders are freed before the DiT loads.

| Implementation | Seconds | Memory, GiB | Processes |
| --- | ---: | ---: | ---: |
{{TEXT_MLX}}
{{TEXT_8BIT}}
{{TEXT_TORCH}}

Torch pads the prompt to 1153 tokens as the diffusers pipeline does; MLX runs the real tokens only, with the same result
for them.

### 864×480, 121 frames, 10 steps

Cold, 3 processes each, without loading. All three runs agree, but 3 cannot reach the test's threshold, so this is not a tested
claim. MLX's default cache is used; see the table above for limits.

| Implementation | Denoise, s | Seconds per step | Video decode, s | Audio, s | Total, s | Memory, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLX | 718 | 72.0 | 215 | 1.7 | 935 | 36.5 |
| Torch MPS | 798 | 80.0 | 225 | 1.3 | 1024 | 38.5 |

## Quick start

macOS on Apple Silicon, Python 3.11+, [uv](https://docs.astral.sh/uv/). About 25 GB of disk. Tested on 48 GB only.

```bash
git clone https://github.com/vladislav-vasilenko/kandinsky6-mlx && cd kandinsky6-mlx
uv sync --extra text

# This repository: DiT, video VAE, audio VAE, vocoder (8.5 GB)
uv run hf download VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX --exclude "examples/*" --local-dir weights/lite-distill

# Text encoders from the original repository: Qwen2.5-VL-7B (15.5 GB), CLIP-L, tokenizers
uv run hf download kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers --revision 4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81 \
    --include "text_encoder/*" "tokenizer/*" "text_encoder_2/*" "tokenizer_2/*" --local-dir weights/lite-distill-text
```

Text to video, 1 s at 432×256:

```bash
uv run --extra text python scripts/generate.py --weights weights/lite-distill --preset smoke --seed 42 --out fox.mp4 \
    --prompt "A red fox runs through fresh snow at dawn, slow motion. <AUDCAP>Crunching snow, distant crows.<ENDAUDCAP>"
```

Image to video, and 5 s at 864×480:

```bash
uv run --extra text python scripts/generate.py --weights weights/lite-distill --preset smoke --image input.png \
    --prompt "..." --out from_image.mp4
uv run --extra text python scripts/generate.py --weights weights/lite-distill --preset baseline --prompt "..." --out long.mp4
```

Lower memory: add `--cache-limit-gib 0` (or 4) and `--qwen-bits 8`. Options: `python scripts/generate.py --help`.

From Python:

```python
from kandinsky6_mlx.pipelines.export import write_mp4
from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline
from kandinsky6_mlx.text.encode import TextEncoders

text = TextEncoders.from_files("weights/lite-distill-text").encode_prompt("A red fox ... <AUDCAP>...<ENDAUDCAP>")
pipe = Kandinsky6Pipeline.from_converted("weights/lite-distill")
out = pipe(text, height=256, width=432, num_frames=25, seed=42)
write_mp4("fox.mp4", out.frames[0], out.frame_rate, out.audio[0], out.sample_rate)
```

Prompts work best with the structure Kandinsky 6 expects: style, scene, lighting, camera, initial state, timed actions, and
sound inside `<AUDCAP>…<ENDAUDCAP>`. The seed drives NumPy noise, so it does not reproduce a diffusers or ComfyUI run with
the same seed.

## Verification

Every component is compared with the PyTorch reference (`diffusers==0.42.1`) on the real weights. Full reports:
[`reports/`](https://github.com/vladislav-vasilenko/kandinsky6-mlx/tree/main/reports).

| Component | Check | Result |
| --- | --- | --- |
| Whole pipeline, float32 DiT | 10 steps vs diffusers from the same noise, 4 seeds | latents rel_l2 ≤ 7.2e-5, frames ≥ 96.6 dB, audio ≥ 84.2 dB |
| Whole pipeline, bf16 DiT | vs the float32 reference, relative to Torch bf16 | rel_l2 within 1.49× of Torch bf16's drift; frames 27–46 dB, Torch 27–37 dB |
| Image to video | 4 cases, float32 and bf16 | float32: rel_l2 ≤ 1.1e-5, frames ≥ 110 dB; bf16: step cosine ≥ 0.9996, frames 42–50 dB |
| DiT, bf16 | every block and final outputs | 0.56–1.12× Torch bf16's error, cosine ≥ 0.99997 per block |
| Video VAE decoder, float32 | 25 frames | 121.7 dB |
| Tiled VAE decoding | first 29 of 121 frames at 864×480 | 130.4 dB. The whole clip's reference needs 40 GB and has not been recorded |
| VAE encoder and image preprocessing | fixed images | preprocessing bit-identical to diffusers; encoder rel_l2 ≤ 1e-3 |
| Audio VAE + vocoder, float32 | waveform | SNR 92.9 dB |
| Qwen2.5-VL (mlx-vlm), float32 | hidden states, 5 prompts | rel_l2 ≤ 2.3e-5 |
| Qwen2.5-VL, bf16 | same | rel_l2 within 1.17× of Torch bf16's drift; token cosine ≥ 0.9985 (Torch: 0.9984) |
| CLIP-L, float32 | pooled output | rel_l2 3.9e-6 |
| PiFlow scheduler | 10 steps | bit-identical |

## Files

| File | Tensors | Dtypes | Size | SHA-256 |
| --- | ---: | --- | ---: | --- |
| `transformer/model.safetensors` | 2399 | 1976 BF16, 423 F32 | 5.94 GiB | `62a3dba4a4187f56e525997f0270b282e3787f735879b0bc3cd316dbc4579fee` |
| `vae/model.safetensors` | 248 | F32 | 0.92 GiB | `c90f898a5284ed365630f02d099247b6c0ae79ab6c93d16c0fbcbb109c693d12` |
| `audio_vae/model.safetensors` | 59 | F32 | 1.14 GiB | `ad3869e1145db1e5ce2f657c909ede6586278655e00960f58f9507b0b2d890a3` |
| `vocoder/model.safetensors` | 667 | F32 | 0.46 GiB | `7d903ad9b4c3d80cef04b5fbcca30568e3c6485102b32493e4f9dec0ae09f388` |

The conversion moves convolution weights to MLX's channels-last layouts and drops the duplicate vocoder stored inside the
original `audio_vae`. Each tensor keeps its stored dtype. As in diffusers, the DiT's time embeddings and modulation run in
float32. Each folder keeps the original `config.json` and a `conversion.json` with source hashes. `examples/` holds the clips,
posters and `prompts.json`. Text encoders are not included: `text_encoder/` of the original repository is the unmodified
Qwen2.5-VL-7B-Instruct, and `text_encoder_2/` is CLIP-L.

## Quality and limits

- Tested on one machine, an M3 Max with 48 GB. Smaller Macs are untested, including the memory settings above.
- The measurements used a single-file copy of Qwen2.5-VL, checked against the original `text_encoder/` shards (all 729
  tensors' names, shapes and dtypes, and their bytes in part). The quick start's download of the shards has not itself been run.
- Visual checks cover a handful of prompts. No perceptual metric, such as VBench, has been run.
- Image-to-video is verified at 432×256. At 864×480 with tiling, diffusers would also encode the image in tiles; this port
  encodes it in one pass, so results there can differ.
- The non-distilled Lite-5s (FlowMatch, guidance, 50 steps) runs in the same pipeline and is verified, but its weights are not
  in this repository. Pro models are untested.

## License

MIT, as the [original model](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers). The separately
downloaded Qwen2.5-VL-7B-Instruct: Apache 2.0.

## Citation

Please cite the original work:

```bibtex
@misc{kandinsky2026kandinsky60videofoundation,
      title={Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation},
      author={Team Kandinsky and Julia Agafonova and Bulat Akhmatov and Mikhail Aksyutin and Grigorii Alekseenko and Anastasia Aliaskina and Olga Androsova and Vladimir Arkhipkin and Anna Averchenkova and Alexander Belykh and Serafima Bocharova and Sofiya Bogakovskaya and Anton Bukashkin and Mark Bulygin and Kirill Buzygin and Irina Cheremnykh and Kirill Chernyshev and Mikhail Chernyshov and Vladimir Chernyy and David Chikovani and Georgy Daniltsev and Denis Dimitrov and Anna Dmitrienko and Vladimir Dokholyan and Sergey Emelyanov and Dmitry Ermilov and Georgii Fedorov and Polina Gavrilova and Nikolai Gerasimenko and Aleksandr Gordeev and Andrey Inozemtsev and Andrei Ivaniuta and Alexander Ivanov and Mikhail Karaev and Anastasiia Kargapoltseva and Ivan Kirillov and Nikita Kiselev and Valeria Kobenko and Yury Kolabushin and Denis Koposov and Anatoly Korobov and Vladimir Korviakov and Kirill Kozlov and Denis Krzhivokolskiy and Konstantin Kuklev and Alexander Kunitsyn and Sergey Kuzin and Vladislav Lakhtionov and Alexey Letunovskiy and Maxim Litvinov and Alexander Lyulkov and Georgy Makarov and Kirill Malakhov and Egor Malykh and Mikhail Mamaev and Dmitrii Mikhailov and Polina Mikhailova and Ivan Mikheev and Elizaveta Muromtseva and Nikolai Nazarkin and Tatiana Nikulina and Lev Novitskiy and Stanislav Onuchin and Nikita Osterov and Denis Parkhomenko and Anatoliy Parpara and Vladimir Polovnikov and Konstantin Reznikov and Azat Saginbaev and Nikita Samsonov and Alexander Sentsov and Nikita Shaimov and Artem Sherstyuk and Andrey Shutkin and Egor Silvestrov and Bulat Suleimanov and Matvey Suprunov and Sergey Taranov and Irina Tolstykh and Tatiana Trofimuk and Ilya Trushkin and Aleksandra Tsybina and Olga Varlashina and Viacheslav Vasilev and Ilya Vasiliev and Eugeny Vilisov and Sergey Yakubson and Konstantin Zakharov},
      year={2026},
      eprint={2610.05608},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.05608},
}
```
