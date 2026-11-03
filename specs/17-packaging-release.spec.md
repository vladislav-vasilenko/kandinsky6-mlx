---
id: 17-packaging-release
title: CLI, documentation, converted weights and first release
depends_on: [09-pipeline-t2v-mvp, 11-text-encoders, 12-i2v, 13-flowmatch-cfg]
estimate: 3-4 days (1-2 for packaging, 2 for ComfyUI)
real_weights: required
---

## Goal

Others can install the package, download converted weights and generate a video with one command, or from ComfyUI with nodes and workflows shipped in this repository.

## Deliverables

- `kandinsky6-mlx` console script (`generate`, `convert`).
- README: install, hardware table (from spec 14), examples, supported checkpoints and their validation status, known limits, links to the original model cards and licenses.
- Model cards for converted weights: source repo and revision, conversion command, SHA-256, parity metrics from reports, license (Kandinsky 6: MIT; Qwen2.5-VL and CLIP: their own licenses, not redistributed unless permitted).
- Upload script `scripts/upload_weights.py` (dry-run by default).
- GitHub release notes and a version tag.
- ComfyUI node pack `comfyui/` (see below), with workflows in `comfyui/workflows/`.
- `reports/17-packaging-release.json`.

## ComfyUI nodes and workflows

Separate nodes for each stage of the MLX pipeline, so a workflow shows the stages and can reuse their results:

| Node | Input | Output |
|---|---|---|
| Load models | weights folder, precision, memory options (cache limit, tiling) | MLX pipeline handle |
| Encode text | handle, prompt | text embeddings |
| Sample (PiFlow or FlowMatch + CFG) | handle, embeddings, size, frames, seed, steps, optional image for I2V | video and audio latents |
| Decode video | handle, latents | ComfyUI `IMAGE` |
| Decode audio | handle, latents | ComfyUI `AUDIO` |

Workflows: Lite-distill T2V, Lite-distill I2V, Lite-5s T2V with CFG. Each states the checkpoint and the memory it needs, from the spec 14 numbers.

Implementation notes:

- **Latents and embeddings stay MLX arrays inside custom ComfyUI types.** Convert to torch CPU tensors only at the `IMAGE` and `AUDIO` outputs, in ComfyUI's layouts (`[F, H, W, C]` float in 0..1; `{"waveform": [B, C, S], "sample_rate": ...}`).
- **ComfyUI's model manager does not see MLX memory.** Before sampling, ask it to unload its own models and empty the MPS cache; release the DiT before decoding and clear MLX's cache afterwards, as the pipeline does. Measure the footprint, including what ComfyUI itself keeps.
- **Seeds use the pipeline's NumPy noise**, not ComfyUI's generator, so a node run matches the CLI run.
- The stages are currently inside `Kandinsky6Pipeline.__call__`. Splitting out denoise and decode must keep the public API and its tests.
- ComfyUI's Python environment must take `mlx` and `mlx-vlm==0.7.11`. Check for dependency conflicts with ComfyUI's own pins first, because `mlx-vlm` needs `transformers`.
- The pack imports MLX lazily and reports a clear error off Apple Silicon.

## Rules

- Uploading weights and publishing a release are outward actions: prepare everything with `--dry-run`, then wait for the owner's explicit approval.
- Follow the Attribution section of `AGENTS.md` in every public text.
- Claim only what reports prove: e.g. "parity with diffusers 0.42.1 on Lite-distill Smoke (cosine ≥ 0.999 per step)", not "identical output".

## Tests

- Fresh virtual environment: `pip install` from the built wheel, `kandinsky6-mlx generate --help` works.
- Upload script dry run lists the files and sizes it would upload.
- Nodes on tiny fixtures, without ComfyUI installed: the node chain gives the same frames and audio as the pipeline for the same seed; output layouts and value ranges are ComfyUI's.
- Every shipped workflow is valid JSON and uses only existing node names and inputs (checked against `INPUT_TYPES`).

## Acceptance criteria

- [ ] Wheel builds; clean install works.
- [ ] Model card and README state validation status per checkpoint, consistent with reports.
- [ ] Dry runs prepared; publication waits for approval.
- [ ] Each shipped workflow ran in a real ComfyUI on real weights at Smoke; frames match the CLI for the same seed (PSNR >= 40 dB); peak footprint and time recorded next to the CLI's.
- [ ] ComfyUI version and installation steps documented in the README.

## Out of scope

ComfyUI's own samplers, `MODEL` and `LATENT` types and KSampler compatibility, LoRA, Pro. Publishing the pack in a ComfyUI registry is an outward action and needs the owner's approval.
