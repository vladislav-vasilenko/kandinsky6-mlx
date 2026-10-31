"""Spec 13: FlowMatchEulerDiscreteScheduler in MLX against diffusers 0.42.1, released Lite-5s configuration."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from kandinsky6_mlx.utils.compare import compare

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def reference(config=CONFIG):
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler as Reference

    return Reference.from_config({k: v for k, v in config.items() if k != "_class_name"})


def test_config_matches_the_released_lite_5s_scheduler():
    released = ROOT / "weights" / "lite-base" / "scheduler" / "scheduler_config.json"
    if not released.is_file():
        pytest.skip(f"needs {released}")
    config = json.loads(released.read_text())
    assert {k: v for k, v in config.items() if not k.startswith("_")} == {
        k: v for k, v in CONFIG.items() if not k.startswith("_")
    }


@pytest.mark.parametrize("steps", [1, 7, 10, 50])
def test_timesteps_and_sigmas(steps):
    ref = reference()
    ref.set_timesteps(steps)
    ours = FlowMatchEulerDiscreteScheduler.from_config(CONFIG)
    ours.set_timesteps(steps)
    np.testing.assert_array_equal(np.array(ours.timesteps), ref.timesteps.numpy())
    np.testing.assert_array_equal(np.array(ours.sigmas), ref.sigmas.numpy())


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_fifty_steps(dtype):
    """50 Euler steps with random model outputs; bf16 follows torch's bf16 arithmetic of `dt * model_output`."""
    torch = pytest.importorskip("torch")
    tdtype, mdtype = {"float32": (torch.float32, mx.float32), "bfloat16": (torch.bfloat16, mx.bfloat16)}[dtype]
    ref = reference()
    ref.set_timesteps(50)
    ours = FlowMatchEulerDiscreteScheduler.from_config(CONFIG)
    ours.set_timesteps(50)
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 3, 4, 6, 16)).astype(np.float32)
    ref_sample = torch.from_numpy(sample).to(tdtype)
    our_sample = mx.array(sample).astype(mdtype)
    for t_ref, t_ours in zip(ref.timesteps, np.array(ours.timesteps)):
        out = rng.standard_normal(sample.shape).astype(np.float32)
        ref_sample = ref.step(torch.from_numpy(out).to(tdtype), t_ref, ref_sample, return_dict=False)[0]
        our_sample = ours.step(mx.array(out).astype(mdtype), mx.array(np.array([t_ours])), our_sample)
    assert our_sample.dtype == mdtype
    limit = 1e-5 if dtype == "float32" else 0.0
    assert compare(our_sample.astype(mx.float32), ref_sample.float())["rel_max"] <= limit


def test_unsupported_options_are_refused():
    with pytest.raises(NotImplementedError):
        FlowMatchEulerDiscreteScheduler.from_config(CONFIG | {"use_dynamic_shifting": True})
