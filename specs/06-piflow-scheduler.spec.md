---
id: 06-piflow-scheduler
title: PiFlow scheduler for distilled checkpoints
depends_on: [01-reference-harness]
estimate: 1.5 days
real_weights: none
---

## Goal

An MLX `PiflowScheduler` that reproduces `set_timesteps` and every `step` of the diffusers scheduler for the distilled checkpoints (Lite-distill, Pro-distill).

## Read first

- `diffusers/schedulers/scheduling_piflow.py` (0.42.1): the policy class (`_interpolate` 72, `_unwarp_t` 83, `pi` 86) and `PiflowScheduler` (`set_timesteps` 223, `_to_grid` 255, `_policy_step` 273, `_step_index_for` 304, `step` 324).
- Released `scheduler/scheduler_config.json` (Lite-distill): `n_grid=10`, `num_policy_substeps=128`, `shift=5.0`, `final_step_size_scale=0.5`, `num_train_timesteps=1000`, `use_dynamic_shifting=false`, `eps=1e-6`.
- How the pipeline calls it: `pipeline_kandinsky6_ti2va.py` `__call__` (646–) and `retrieve_timesteps` (105).

## Deliverables

- `src/kandinsky6_mlx/schedulers/piflow.py` (`PiflowScheduler.from_config(dict)`, `set_timesteps(n)`, `step(model_output, timestep, sample)`).
- `tests/parity/test_piflow.py`.
- `reports/06-piflow-scheduler.json`.

## Implementation notes

- The scheduler is pure math over tensors; compute in float32 regardless of model dtype, as the reference does.
- `num_policy_substeps=128` inner substeps per outer step: make sure the loop count, the grid interpolation and the final-step scale match; a wrong count still produces plausible latents.
- Keep the scheduler stateless enough that two pipelines can run in one process (no module-level state).

## Tests

- `set_timesteps(10)` timesteps and sigmas equal the reference (`rel_max <= 1e-6`).
- Ten consecutive `step` calls on a fixed sequence of NumPy model outputs (fixture from spec 01) give the reference samples (`rel_max <= 1e-5`), for latent shapes of video `[1,16,7,32,54]`-like and audio `[1,45,40]`-like tensors.
- Determinism: `stochastic_sampling=false` yields identical results across runs.

## Validation runs

```bash
uv run --group reference pytest tests/parity/test_piflow.py -q
```

## Acceptance criteria

- [ ] Timesteps and every step output match at the tolerances above.
- [ ] Report lists measured `rel_max` per step.

## Out of scope

FlowMatchEulerDiscrete (spec 13).

## Stop conditions

Reference depends on torch-specific behavior that cannot be reproduced (e.g. an integer rounding rule) → document exactly and report `FAIL` rather than approximating silently.
