"""Spec 06: MLX PiFlow scheduler against diffusers 0.42.1 `PiflowScheduler`."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.schedulers.piflow import PiflowScheduler
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture
from tests.parity.dit_cases import TINY

CONFIG = json.loads((Path(__file__).resolve().parents[2] / "fixtures/configs/lite-distill-scheduler.json").read_text())
TENSORS, META = load_fixture(TINY / "piflow", backend="mlx")
# Smoke-size latents in the transformer layout: video (B, T, H, W, C), audio (B, L, C).
SMOKE_SHAPES = {"video": (1, 7, 32, 54, 16), "audio": (1, 45, 40)}


def run_mlx(scheduler: PiflowScheduler, sample, outputs) -> list[mx.array]:
    samples = []
    for timestep, output in zip(np.array(scheduler.timesteps), outputs):
        sample = scheduler.step(mx.array(output), mx.array(timestep), sample)
        samples.append(sample)
    return samples


def run_torch(config: dict, steps: int, sample: np.ndarray, outputs, dtype=None) -> tuple:
    torch = pytest.importorskip("torch")
    from diffusers import PiflowScheduler as Reference

    dtype = dtype or torch.float32
    scheduler = Reference.from_config(config)
    scheduler.set_timesteps(steps)
    x = torch.from_numpy(sample).to(dtype)
    samples = []
    for timestep, output in zip(scheduler.timesteps, outputs):
        x = scheduler.step(torch.from_numpy(output).to(dtype), timestep, x, return_dict=False)[0]
        samples.append(x)
    return scheduler, samples


def random_case(kind: str, steps: int, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    shape = SMOKE_SHAPES[kind]
    sample = rng.standard_normal(shape).astype(np.float32)
    widened = (*shape[:-1], shape[-1] * CONFIG["n_grid"])
    return sample, [rng.standard_normal(widened).astype(np.float32) for _ in range(steps)]


def test_released_config_loads():
    scheduler = PiflowScheduler.from_config(CONFIG)
    assert (scheduler.n_grid, scheduler.num_policy_substeps, scheduler.shift) == (10, 128, 5.0)
    assert (scheduler.final_step_size_scale, scheduler.eps, scheduler.num_train_timesteps) == (0.5, 1e-6, 1000)


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_fixture_schedule_and_ten_steps(kind):
    scheduler = PiflowScheduler.from_config(META["config"])
    scheduler.set_timesteps(META["num_inference_steps"])
    assert compare(scheduler.timesteps, TENSORS[f"outputs.{kind}.timesteps"])["rel_max"] <= 1e-6
    assert compare(scheduler.sigmas, TENSORS[f"outputs.{kind}.sigmas"])["rel_max"] <= 1e-6
    outputs = [TENSORS[f"inputs.{kind}.model_output.{i}"] for i in range(META["num_inference_steps"])]
    samples = run_mlx(scheduler, TENSORS[f"inputs.{kind}.sample"], outputs)
    for i, sample in enumerate(samples):
        assert compare(sample, TENSORS[f"outputs.{kind}.sample.{i}"])["rel_max"] <= 1e-5, i


@pytest.mark.parametrize("steps", [1, 4, 10])
@pytest.mark.parametrize("kind", ["video", "audio"])
def test_smoke_shapes_match_torch(kind, steps):
    sample, outputs = random_case(kind, steps)
    reference, ref_samples = run_torch(CONFIG, steps, sample, outputs)
    scheduler = PiflowScheduler.from_config(CONFIG)
    scheduler.set_timesteps(steps)
    assert np.array_equal(np.array(scheduler.timesteps), reference.timesteps.numpy())
    assert np.array_equal(np.array(scheduler.sigmas), reference.sigmas.numpy())
    for i, (got, ref) in enumerate(zip(run_mlx(scheduler, mx.array(sample), outputs), ref_samples)):
        assert compare(got, ref)["rel_max"] <= 1e-5, i


def test_bf16_sample_and_model_output():
    """The update runs in float32 and returns the sample dtype, as in the reference."""
    torch = pytest.importorskip("torch")
    sample, outputs = random_case("audio", 10)
    _, ref_samples = run_torch(CONFIG, 10, sample, outputs, dtype=torch.bfloat16)
    scheduler = PiflowScheduler.from_config(CONFIG)
    scheduler.set_timesteps(10)
    x = mx.array(sample).astype(mx.bfloat16)
    for i, (timestep, output) in enumerate(zip(np.array(scheduler.timesteps), outputs)):
        x = scheduler.step(mx.array(output).astype(mx.bfloat16), mx.array(timestep), x)
        assert x.dtype == mx.bfloat16
        # Inputs are bit-identical bf16 and the update is float32, so outputs agree up to one bf16 rounding.
        assert compare(x, ref_samples[i])["rel_max"] <= 1e-2, i


@pytest.mark.parametrize("steps", [1, 4, 10])
def test_substep_counts_match_reference_formula(steps):
    """Loop count per step, from the reference's own raw schedule: `round(delta * 128).long().clamp(min=1)`."""
    torch = pytest.importorskip("torch")
    from diffusers import PiflowScheduler as Reference

    reference = Reference.from_config(CONFIG)
    reference.set_timesteps(steps)
    raw = reference._piflow_raw_timesteps
    ends = torch.cat([raw[1:], torch.tensor([reference.eps])])
    expected = ((raw - ends) * reference.num_policy_substeps).round().to(torch.long).clamp(min=1).tolist()
    scheduler = PiflowScheduler.from_config(CONFIG)
    scheduler.set_timesteps(steps)
    assert [scheduler.num_substeps(i) for i in range(steps)] == expected


def test_wrong_substep_count_is_detected():
    """Fewer substeps give plausible latents; the step tolerance must still reject them. (127 instead of 128 gives the
    same rounded counts on this schedule, so the comparison uses 64.)"""
    sample, outputs = random_case("audio", 10)
    _, ref_samples = run_torch(CONFIG, 10, sample, outputs)
    scheduler = PiflowScheduler.from_config(CONFIG | {"num_policy_substeps": 64})
    scheduler.set_timesteps(10)
    final = run_mlx(scheduler, mx.array(sample), outputs)[-1]
    assert compare(final, ref_samples[-1])["rel_max"] > 1e-5


def test_deterministic_and_independent_instances():
    """Identical runs give identical bits; two schedulers stepped in alternation do not share state."""
    sample, outputs = random_case("audio", 10)
    first, second = PiflowScheduler.from_config(CONFIG), PiflowScheduler.from_config(CONFIG)
    first.set_timesteps(10)
    second.set_timesteps(10)
    x, y = mx.array(sample), mx.array(sample)
    for timestep, output in zip(np.array(first.timesteps), outputs):
        x = first.step(mx.array(output), mx.array(timestep), x)
        y = second.step(mx.array(output), mx.array(timestep), y)
    assert np.array_equal(np.array(x), np.array(y))


def test_step_index_rules():
    scheduler = PiflowScheduler.from_config(CONFIG)
    scheduler.set_timesteps(3)
    sample, outputs = random_case("audio", 3)
    with pytest.raises(ValueError, match="integer"):
        scheduler.step(mx.array(outputs[0]), 1, mx.array(sample))
    with pytest.raises(ValueError, match="not in the Piflow schedule"):
        scheduler.step(mx.array(outputs[0]), mx.array(123.0), mx.array(sample))
    scheduler.set_begin_index(2)
    scheduler.step(mx.array(outputs[0]), mx.array(0.0), mx.array(sample))
    assert scheduler.step_index == 3
    with pytest.raises(RuntimeError, match="exhausted"):
        scheduler.step(mx.array(outputs[0]), mx.array(0.0), mx.array(sample))


def test_output_width_is_checked():
    scheduler = PiflowScheduler.from_config(CONFIG)
    scheduler.set_timesteps(2)
    with pytest.raises(ValueError, match="channels"):
        scheduler.step(mx.zeros((1, 4, 39)), scheduler.timesteps[0], mx.zeros((1, 4, 4)))
