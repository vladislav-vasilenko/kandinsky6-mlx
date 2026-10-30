---
id: 13-flowmatch-cfg
title: Non-distilled checkpoints — FlowMatch Euler scheduler and classifier-free guidance
depends_on: [09-pipeline-t2v-mvp, 11-text-encoders]
estimate: 2 days
real_weights: stage-b
---

## Goal

Support Lite-5s, Lite-pretrain-5s (and, by config, Pro-5s, Pro-pretrain-5s): `FlowMatchEulerDiscreteScheduler`, negative prompt, CFG with `guidance_scale=5`, 50 steps.

## Read first

- `diffusers/schedulers/scheduling_flow_match_euler_discrete.py` (`set_timesteps`, `step`, `shift=5.0`, `time_shift_type="exponential"`, `use_dynamic_shifting=false`).
- `pipeline_kandinsky6_ti2va.py`: `do_classifier_free_guidance`, negative prompt encoding, how conditional and unconditional predictions are combined (batched or two passes), default negative prompt.

## Deliverables

- `src/kandinsky6_mlx/schedulers/flow_match_euler.py`; scheduler selection from `scheduler_config.json` `_class_name`.
- CFG in the pipeline; option to batch cond/uncond (B=2) or run sequentially (lower memory).
- `tests/parity/test_flow_match.py`, `tests/e2e/test_cfg_tiny.py`, `tests/e2e/test_lite_base_real.py`.
- `reports/13-flowmatch-cfg.json`.

## Tests

- Scheduler: timesteps/sigmas and 50 steps parity (`rel_max <= 1e-6` / `1e-5`).
- CFG tiny e2e: batched and sequential modes equal each other and the reference.
- Stage B: Lite-5s Smoke, 50 steps, guidance 5 — per-step cosine and final PSNR as in spec 09; wall time vs Lite-distill.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_flow_match.py tests/e2e/test_cfg_tiny.py -q
uv run python scripts/generate.py --weights weights/lite-base --prompt "..." --preset smoke --out outputs/13/lite_base.mp4
```

## Acceptance criteria

- [ ] Scheduler and CFG parity.
- [ ] Stage B criteria met for Lite-5s (needs its converted weights, about 6 GB DiT).

## Out of scope

MagCache and other step-skipping accelerations (native runtime only); a follow-up may port them.

## Stop conditions

Reference pipeline uses guidance rescale or other terms not listed here → implement and document them.
