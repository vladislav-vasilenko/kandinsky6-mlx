---
id: 15-packaging-release
title: CLI, documentation, converted weights and first release
depends_on: [09-pipeline-t2v-mvp, 11-text-encoders]
estimate: 1-2 days
real_weights: required
---

## Goal

Others can install the package, download converted weights and generate a video with one command.

## Deliverables

- `kandinsky6-mlx` console script (`generate`, `convert`).
- README: install, hardware table (from spec 14), examples, supported checkpoints and their validation status, known limits, links to the original model cards and licenses.
- Model cards for converted weights: source repo and revision, conversion command, SHA-256, parity metrics from reports, license (Kandinsky 6: MIT; Qwen2.5-VL and CLIP: their own licenses, not redistributed unless permitted).
- Upload script `scripts/upload_weights.py` (dry-run by default).
- GitHub release notes and a version tag.
- `reports/15-packaging-release.json`.

## Rules

- Uploading weights and publishing a release are outward actions: prepare everything with `--dry-run`, then wait for the owner's explicit approval.
- Follow the Attribution section of `AGENTS.md` in every public text.
- Claim only what reports prove: e.g. "parity with diffusers 0.42.1 on Lite-distill Smoke (cosine ≥ 0.999 per step)", not "identical output".

## Tests

- Fresh virtual environment: `pip install` from the built wheel, `kandinsky6-mlx generate --help` works.
- Upload script dry run lists the files and sizes it would upload.

## Acceptance criteria

- [ ] Wheel builds; clean install works.
- [ ] Model card and README state validation status per checkpoint, consistent with reports.
- [ ] Dry runs prepared; publication waits for approval.

## Out of scope

ComfyUI integration (follow-up).
