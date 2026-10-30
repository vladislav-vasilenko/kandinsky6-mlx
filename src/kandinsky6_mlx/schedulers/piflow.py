"""PiFlow scheduler for distilled Kandinsky 6 checkpoints, in MLX.

Mirrors `diffusers/schedulers/scheduling_piflow.py` (diffusers 0.42.1). The model is evaluated once per step and returns
`n_grid` predictions per channel; between evaluations a network-free policy interpolates those predictions and is
integrated with `num_policy_substeps` Euler substeps per unit of raw time.

The per-substep scalars (raw times, sigmas, interpolation indices and weights, substep counts) depend only on the
schedule, so they are computed on the host in NumPy float32, which applies the same IEEE operations as the reference's
float32 torch scalars (including round-half-to-even for the substep count). Tensor updates run in MLX in float32.
"""

import mlx.core as mx
import numpy as np


def shift_timesteps(t, shift: float):
    """Map raw flow-matching time to the shifted DiT time."""
    return shift * t / (1 + (shift - 1) * t)


def unwarp_timesteps(sigma, shift: float):
    """Inverse of `shift_timesteps`."""
    return sigma / (shift + (1 - shift) * sigma)


class PiflowScheduler:
    """Few-step PiFlow scheduler for widened-output diffusion transformers.

    Holds only per-instance state (the schedule and the step index), so independent pipelines can each own one.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        n_grid: int = 10,
        eps: float = 1e-6,
        final_step_size_scale: float = 0.5,
        num_policy_substeps: int = 128,
    ):
        if n_grid < 2:
            raise ValueError(f"PiflowScheduler requires n_grid >= 2, got {n_grid}")
        if eps <= 0:
            raise ValueError(f"PiflowScheduler requires eps > 0, got {eps}")
        if not 0 < final_step_size_scale <= 1:
            raise ValueError("PiflowScheduler requires 0 < final_step_size_scale <= 1")
        if num_policy_substeps < 1:
            raise ValueError("PiflowScheduler requires num_policy_substeps >= 1")
        self.num_train_timesteps = num_train_timesteps
        self.shift = float(shift)
        self.n_grid = int(n_grid)
        self.eps = float(eps)
        self.final_step_size_scale = float(final_step_size_scale)
        self.num_policy_substeps = int(num_policy_substeps)
        self.num_inference_steps: int | None = None
        self.timesteps = mx.array([], dtype=mx.float32)
        self.sigmas = mx.array([], dtype=mx.float32)
        self._raw = np.zeros(0, np.float32)
        self._sigmas = np.zeros(0, np.float32)
        self._timesteps = np.zeros(0, np.float32)
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @classmethod
    def from_config(cls, config: dict) -> "PiflowScheduler":
        """Build from `scheduler_config.json`; diffusers bookkeeping keys (`_class_name`, ...) are ignored."""
        names = ("num_train_timesteps", "shift", "n_grid", "eps", "final_step_size_scale", "num_policy_substeps")
        return cls(**{key: config[key] for key in names if key in config})

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the distilled schedule: `num_inference_steps` raw times ending one scaled segment before 0."""
        if num_inference_steps is None or num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        segment = 1.0 / (num_inference_steps - (1.0 - self.final_step_size_scale))
        # float32 tensor times a Python float stays float32 in torch and in NumPy (weak scalars).
        raw = 1.0 - np.arange(num_inference_steps, dtype=np.float32) * np.float32(segment)
        sigmas = shift_timesteps(raw, self.shift)
        self.num_inference_steps = int(num_inference_steps)
        self._raw = raw
        self._sigmas = np.concatenate([sigmas, np.zeros(1, np.float32)])
        self._timesteps = sigmas * np.float32(self.num_train_timesteps)
        self.timesteps = mx.array(self._timesteps)
        self.sigmas = mx.array(self._sigmas)
        self._step_index = None
        self._begin_index = None

    def _to_grid(self, model_output: mx.array, sample: mx.array) -> mx.array:
        """`(..., n_grid * C)` -> `(B, n_grid, ..., C)`."""
        if model_output.ndim != sample.ndim or model_output.shape[:-1] != sample.shape[:-1]:
            raise ValueError(
                "Piflow model output must match sample shape except for the output channels: "
                f"got {tuple(model_output.shape)} for sample {tuple(sample.shape)}"
            )
        if model_output.shape[-1] != sample.shape[-1] * self.n_grid:
            raise ValueError(
                "Piflow model output channels do not match the sample: "
                f"expected {sample.shape[-1] * self.n_grid}, got {model_output.shape[-1]}"
            )
        grid = model_output.reshape(*model_output.shape[:-1], self.n_grid, sample.shape[-1])
        return mx.moveaxis(grid, -2, 1)

    def _step_index_for(self, timestep) -> int:
        if self._step_index is None:
            if self._begin_index is not None:
                self._step_index = self._begin_index
            else:
                value = np.float32(np.asarray(timestep, dtype=np.float32).reshape(-1)[0])
                indices = np.flatnonzero(self._timesteps == value)
                if not indices.size:
                    raise ValueError(f"timestep {float(value)} is not in the Piflow schedule")
                self._step_index = int(indices[1 if indices.size > 1 else 0])
        if self._step_index >= self.num_inference_steps:
            raise RuntimeError("PiflowScheduler.step called after the schedule was exhausted")
        return self._step_index

    def _segment(self, step_index: int) -> tuple[np.float32, np.float32]:
        """Raw times `(src, dst)` of a step; the last step ends at `eps` instead of 0."""
        raw_dst = self._raw[step_index + 1] if step_index + 1 < self._raw.size else np.float32(self.eps)
        return self._raw[step_index], raw_dst

    def num_substeps(self, step_index: int) -> int:
        """Euler substeps of the policy rollout in a step: `round(delta_raw * num_policy_substeps)`, at least 1."""
        raw_src, raw_dst = self._segment(step_index)
        return max(int(np.round((raw_src - raw_dst) * np.float32(self.num_policy_substeps))), 1)

    def _policy_step(self, model_output: mx.array, sample: mx.array, step_index: int) -> mx.array:
        """Integrate the DX policy over one segment; `sample` is float32."""
        grid = self._to_grid(model_output, sample)
        shift, eps = self.shift, np.float32(self.eps)
        raw_src, raw_dst = self._segment(step_index)
        sigma_src = self._sigmas[step_index]

        # DXPolicy: x0 predictions at the grid points spanning [raw_dst, raw_src] of the policy's own segment.
        policy_raw_src = unwarp_timesteps(sigma_src, shift)
        policy_raw_dst = np.maximum(policy_raw_src - (raw_src - raw_dst), np.float32(0))
        policy_segment = np.maximum(policy_raw_src - policy_raw_dst, eps)
        grid_x0 = sample[:, None] - mx.array(sigma_src) * grid

        # policy_rollout_fm
        num_substeps = self.num_substeps(step_index)
        substep_size = (raw_src - raw_dst) / np.float32(num_substeps)
        raw_t, sigma_t, x_t = raw_src, sigma_src, sample
        last = self.n_grid - 1
        for _ in range(num_substeps):
            # pi(x_t, sigma_t): x0 interpolated between neighbouring grid points, velocity (x_t - x0) / sigma_t.
            position = np.clip((unwarp_timesteps(sigma_t, shift) - policy_raw_dst) / policy_segment, 0, 1)
            position = position * np.float32(last)
            t0 = int(np.clip(np.floor(position), 0, last - 1))
            w0, w1 = np.float32(t0 + 1) - position, position - np.float32(t0)
            x_0 = mx.array(w0) * grid_x0[:, t0] + mx.array(w1) * grid_x0[:, t0 + 1]
            velocity = (x_t - x_0) / mx.array(np.maximum(sigma_t, eps))

            raw_t = np.maximum(raw_t - substep_size, np.float32(0))
            sigma_next = shift_timesteps(raw_t, shift)
            x_t = x_t + velocity * mx.array(sigma_next - sigma_t)
            sigma_t = sigma_next
        return x_t

    def step(self, model_output: mx.array, timestep, sample: mx.array) -> mx.array:
        """Advance one step; returns the sample at the next grid point in the dtype of `sample`."""
        if isinstance(timestep, int) or (isinstance(timestep, mx.array) and mx.issubdtype(timestep.dtype, mx.integer)):
            raise ValueError(
                "Passing integer indices as timesteps to PiflowScheduler.step() is not supported; "
                "pass a value from scheduler.timesteps instead"
            )
        step_index = self._step_index_for(timestep)
        updated = self._policy_step(model_output, sample.astype(mx.float32), step_index)
        self._step_index += 1
        return updated.astype(sample.dtype)
