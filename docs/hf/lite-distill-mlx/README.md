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

**Apple M3 Max, 48 GB: about 28 s for a 1 s clip with sound at 432×256, against 37 s for the standard weights in
diffusers on Torch MPS, nearly all of that gap from loading. A 5 s clip at 864×480 takes about 17 minutes in MLX.** See
[Speed and memory](#speed-and-memory) for the breakdown and for how much of this is preliminary.

Works: text-to-video and image-to-video with audio, prompt to MP4 in one command, 432×256 and 864×480.
All components are compared with diffusers on real weights: see [Verification](#verification).

## Video examples: Torch MPS vs MLX

**Standard weights in diffusers on Torch MPS vs this repository in MLX · same prompts, text embeddings, noise and seed ·
432×256 · 25 frames · 24 FPS · 10 PiFlow steps · bf16 DiT · audio.**

The clips show the same scene and differ in details: both sides run the DiT in bf16, and over 10 distilled steps its
rounding differences grow differently in each (peak signal-to-noise between the pairs: 23.5–40.7 dB). Against a float32
reference, MLX bf16 drifts no more than Torch bf16: see [Verification](#verification).

### Fox in the snow

<table width="100%">
<thead><tr><th width="50%">Torch MPS (diffusers)</th><th width="50%">MLX</th></tr></thead>
<tbody><tr><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/fox_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): fox"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/fox_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/fox_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): fox MP4</a></video></td><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/fox_smoke_seed42_mlx-poster.webp" aria-label="MLX: fox"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/fox_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/fox_smoke_seed42_mlx.mp4">MLX: fox MP4</a></video></td></tr></tbody>
</table>

<details>
<summary>Prompt</summary>

A red fox runs through fresh snow at dawn, slow motion. &lt;AUDCAP&gt;Crunching snow, distant crows.&lt;ENDAUDCAP&gt;

</details>

### Fashion film

<table width="100%">
<thead><tr><th width="50%">Torch MPS (diffusers)</th><th width="50%">MLX</th></tr></thead>
<tbody><tr><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): orbit_woman"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): orbit_woman MP4</a></video></td><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_smoke_seed42_mlx-poster.webp" aria-label="MLX: orbit_woman"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_smoke_seed42_mlx.mp4">MLX: orbit_woman MP4</a></video></td></tr></tbody>
</table>

<details>
<summary>Prompt</summary>

Cinematic live-action fashion film, natural skin texture, restrained color grading, realistic cloth physics. An adult woman, about 28 years old, stands in a circular stone courtyard. She wears a deep emerald coat and a cream silk scarf. Archways and foreground columns establish clear depth. Warm late-afternoon side light; stable exposure and soft moving highlights. The camera makes a small, smooth 15-degree clockwise arc; keep her face centered. Do not attempt a full circle in one second. Initial state: The woman faces the camera, both feet planted; hair tied loosely; scarf rests on her right shoulder. One uninterrupted shot, about 1 seconds. From 0 to 0.35 seconds: Start from a frontal three-quarter view. The camera begins its arc, a near column slides slowly across the background. From 0.35 to 1 seconds: Continue the arc smoothly. She turns her eyes toward the lens and the scarf lifts slightly, then begins to settle. Same face and outfit throughout. &lt;AUDCAP&gt;Quiet courtyard air and a soft breeze. A brief silk rustle; no dialogue or music.&lt;ENDAUDCAP&gt;

</details>

### Retrofuturist tram

<table width="100%">
<thead><tr><th width="50%">Torch MPS (diffusers)</th><th width="50%">MLX</th></tr></thead>
<tbody><tr><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/strange_tram_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): strange_tram"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/strange_tram_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/strange_tram_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): strange_tram MP4</a></video></td><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/strange_tram_smoke_seed42_mlx-poster.webp" aria-label="MLX: strange_tram"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/strange_tram_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/strange_tram_smoke_seed42_mlx.mp4">MLX: strange_tram MP4</a></video></td></tr></tbody>
</table>

<details>
<summary>Prompt</summary>

Surreal retrofuturist dark comedy, 1970s science-fiction cinema, tactile practical creature effects, worn brass, painted miniatures, muted teal and amber, subtle film grain. A rain-wet tram stop in an imaginary industrial city. An original pear-shaped, blue-skinned commuter with tiny spectacles and a tan raincoat waits beside a brass ticket machine. He has two arms and two legs. Overcast daylight mixed with warm tungsten light in the ticket machine. A short, slow push-in, without changing the view angle. Initial state: The commuter holds one paper ticket. The ticket slot is shut. Steam drifts behind the stop. One uninterrupted shot, about 1 seconds. From 0 to 0.4 seconds: The brass ticket slot clicks open. The commuter freezes and raises one eyebrow. From 0.4 to 1 seconds: A small mechanical tongue briefly taps the edge of his ticket, then retracts. The commuter blinks in deadpan disbelief. &lt;AUDCAP&gt;One dry metallic click over quiet rain. A soft rubbery squeak and a tiny bell; no speech.&lt;ENDAUDCAP&gt;

</details>

### Biomechanical flower

<table width="100%">
<thead><tr><th width="50%">Torch MPS (diffusers)</th><th width="50%">MLX</th></tr></thead>
<tbody><tr><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/mechanical_bloom_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): mechanical_bloom"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/mechanical_bloom_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/mechanical_bloom_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): mechanical_bloom MP4</a></video></td><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/mechanical_bloom_smoke_seed42_mlx-poster.webp" aria-label="MLX: mechanical_bloom"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/mechanical_bloom_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/mechanical_bloom_smoke_seed42_mlx.mp4">MLX: mechanical_bloom MP4</a></video></td></tr></tbody>
</table>

<details>
<summary>Prompt</summary>

Tactile miniature science-fiction film, stop-motion-inspired creature design with smooth motion, copper mechanisms and translucent botanical tissue, detailed macro photography. An original small flower made of copper ribs and translucent pale-blue petals grows from a cracked ceramic pot on a workbench. The background is an unfocused greenhouse. Soft green backlight through glass with a warm practical lamp on the workbench. A tiny sideways camera slide creating gentle foreground parallax. Initial state: One closed flower bud; a drop of water hangs from a copper leaf. One uninterrupted shot, about 1 seconds. From 0 to 0.5 seconds: The bud quivers and the first pair of petals separates. From 0.5 to 1 seconds: The petals open a little farther and release one small puff of glowing pollen. The pot and bench stay fixed. &lt;AUDCAP&gt;A quiet metallic tick. A soft pneumatic puff and a delicate chime; no voice.&lt;ENDAUDCAP&gt;

</details>

### Scandinavian room

<table width="100%">
<thead><tr><th width="50%">Torch MPS (diffusers)</th><th width="50%">MLX</th></tr></thead>
<tbody><tr><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/room_assembly_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): room_assembly"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/room_assembly_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/room_assembly_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): room_assembly MP4</a></video></td><td width="50%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/room_assembly_smoke_seed42_mlx-poster.webp" aria-label="MLX: room_assembly"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/room_assembly_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/room_assembly_smoke_seed42_mlx.mp4">MLX: room_assembly MP4</a></video></td></tr></tbody>
</table>

<details>
<summary>Prompt</summary>

Photorealistic Scandinavian furniture commercial, clean geometric composition, natural materials, restrained palette. An empty sunlit room with white walls, a pale oak floor and one large window. The room is unoccupied. Soft stable morning light from the left window. Completely locked-off wide shot. Initial state: One unbranded flat-pack cardboard box in the center of the room. No other furniture. One uninterrupted shot, about 1 seconds. From 0 to 0.3 seconds: The sealed cardboard box begins to tremble gently. From 0.3 to 1 seconds: Its top flaps pop open and the first pair of wooden bed rails rises a short distance above the box. The rest stays packed. &lt;AUDCAP&gt;A low brief cardboard rattle. A soft pop and two wooden clacks. The room is otherwise quiet.&lt;ENDAUDCAP&gt;

</details>

### Image to video

The video starts from the input image, with the Fashion film and Retrofuturist tram prompts.

<table width="100%">
<thead><tr><th width="33%">Input image</th><th width="33%">Torch MPS (diffusers)</th><th width="33%">MLX</th></tr></thead>
<tbody><tr><td width="33%"><img src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_input_woman.webp" alt="Input image" width="100%"></td><td width="33%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_woman_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): i2v_woman"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_woman_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_woman_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): i2v_woman MP4</a></video></td><td width="33%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_woman_smoke_seed42_mlx-poster.webp" aria-label="MLX: i2v_woman"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_woman_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_woman_smoke_seed42_mlx.mp4">MLX: i2v_woman MP4</a></video></td></tr></tbody>
</table>

<table width="100%">
<thead><tr><th width="33%">Input image</th><th width="33%">Torch MPS (diffusers)</th><th width="33%">MLX</th></tr></thead>
<tbody><tr><td width="33%"><img src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_input_tram.webp" alt="Input image" width="100%"></td><td width="33%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_tram_smoke_seed42_torch_mps-poster.webp" aria-label="Torch MPS (diffusers): i2v_tram"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_tram_smoke_seed42_torch_mps.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_tram_smoke_seed42_torch_mps.mp4">Torch MPS (diffusers): i2v_tram MP4</a></video></td><td width="33%"><video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_tram_smoke_seed42_mlx-poster.webp" aria-label="MLX: i2v_tram"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_tram_smoke_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/i2v_tram_smoke_seed42_mlx.mp4">MLX: i2v_tram MP4</a></video></td></tr></tbody>
</table>

### Frame comparison

Frame 12 (0.5 s) of the text-to-video and image-to-video clips above. Left: Torch MPS. Right: MLX. Click for full size.

<a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/torch_mps_vs_mlx_frame12.webp"><img src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/torch_mps_vs_mlx_frame12.webp" alt="Frame 12 of each clip: Torch MPS and MLX" width="100%"></a>

### 5 seconds at 864×480

**MLX only · 864×480 · 121 frames · 24 FPS · 10 steps · audio · the Fashion film preset's long prompt.** A Torch MPS pair
at this size will come with the full benchmark.

<video controls playsinline preload="metadata" width="100%" poster="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_baseline_seed42_mlx-poster.webp" aria-label="MLX: 5 s at 864x480"><source src="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_baseline_seed42_mlx.mp4" type="video/mp4"><a href="https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples/orbit_woman_baseline_seed42_mlx.mp4">MLX: 5 s at 864x480 MP4</a></video>

## Speed and memory

**Apple M3 Max, 40-core GPU, 48 GB, macOS 27.0.1, AC power · MLX 0.32.4 · PyTorch 2.15.0, diffusers 0.42.1.**
Both sides use the same inputs and the configuration of the study notebook: VAE tiling on, the DiT freed before decoding.
Torch MPS runs with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`; at its default of 1.7 it went past 50 GiB on this 48 GB Mac
and into swap.

> **Preliminary.** These tables are single cold runs of the clips above, one process per clip, summarized by the median
> over clips. A full benchmark is in progress (10 independent processes per comparison, cold and warm runs, confidence
> intervals, exact tests); its method and commands are in
> [BENCHMARK.md](https://github.com/vladislav-vasilenko/kandinsky6-mlx/blob/main/BENCHMARK.md), and its results will
> replace these numbers. No difference here is yet a tested claim.

### 432×256, 25 frames, 10 steps

Seconds per stage, cold (first run in a fresh process, OS file cache warm). Memory is the process's peak physical
footprint as macOS accounts it, GPU buffers included.

| Mode | Implementation | Load | Denoise | Video decode | Audio | Total | Total without load | Memory, GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-to-video, 5 prompts | Torch MPS (diffusers) | 9.8 | 16.3 | 11.0 | 0.3 | **37.4** | 27.6 | 38.9 |
| Text-to-video, 5 prompts | MLX | 1.3 | 13.4 | 12.8 | 0.5 | **28.0** | 26.7 | 35.8 |
| Image-to-video, 2 images | Torch MPS (diffusers) | 9.4 | 18.9 | 10.7 | 0.3 | **39.6** | 30.2 | 37.8 |
| Image-to-video, 2 images | MLX | 1.3 | 15.4 | 12.8 | 0.5 | **30.3** | 29.0 | 35.8 |

- **Where MLX gains:** loading (about 8 s) and denoising (about 3 s, 18%).
- **Where Torch MPS gains:** video decoding (about 2 s) and audio. So without loading the totals are within a few percent.
- **Memory:** close at the defaults, because each framework keeps freed buffers in a cache up to its limit. Image encoding
  (under 0.5 s on both sides) is included in the totals and not shown as a column.

### What memory buys in MLX

MLX's buffer cache is what fills the memory above, and `--cache-limit-gib` limits it. Single runs, text-to-video at 432×256:

| Cache limit | Video decode, s | Total without load, s | Memory, GiB |
| --- | ---: | ---: | ---: |
| None (default) | 12.6 | 26.4 | 35.8 |
| 4 GiB | 18.6 | 32.5 | 12.9 |
| 0 | 25.3 | 39.8 | 8.4 |

So the pipeline's generation stages fit in about 8.4 GiB at the cost of about 50% more time. Prompt encoding needs more: see
below. Machines with less memory have not been tested, and a smaller Mac's chip will also be slower.

### Prompt encoding

Warm encode of the 238-token Smoke prompt, median over independent processes, with the text encoders loaded (about 3–10 s
cold). The encoders are freed before the DiT loads.

| Implementation | Seconds | Memory, GiB | Processes |
| --- | ---: | ---: | ---: |
| MLX, Qwen bf16 + CLIP | 0.45 | 14.6 | 5 |
| MLX, Qwen 8-bit + CLIP | 0.46 | 11.6 | 5 |
| Torch MPS, Qwen bf16 + CLIP | 1.51 | 18.5 | 5 |

Torch pads the prompt to 1153 tokens as the diffusers pipeline does; MLX runs the real tokens only, with the same result
for them.

### 864×480, 121 frames, 10 steps (MLX, one run)

Denoising 794 s (79 s per step), video decoding 225 s in 165 tiles, audio 1.6 s, about 17 minutes in total. The OS footprint
at this size will come with the full benchmark.

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
