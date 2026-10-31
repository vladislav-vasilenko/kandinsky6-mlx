"""`FlowMatchEulerDiscreteScheduler` (diffusers 0.42.1) for the non-distilled Kandinsky 6 checkpoints, in MLX.

Covers the released configuration (`shift=5`, no dynamic shifting, no Karras/exponential/beta sigmas, no terminal
stretch, not inverted, deterministic Euler steps) and refuses the others. The schedule is computed exactly as the
reference does: the init sigmas in float32 (which fix `sigma_max` and `sigma_min`), `np.linspace` of timesteps in
float64, the shift in float64, then float32; timesteps are `sigmas * 1000`, and a final sigma of 0 is appended.

`step` follows the reference's dtypes: the sample is upcast to float32 and `dt * model_output` is computed in the
model output's dtype. Torch keeps a bf16 tensor times a 0-d float32 tensor in bf16 and casts the scalar `dt` to bf16
first (checked bit for bit); the product is added to the float32 sample and the result cast back.
"""

import mlx.core as mx
import numpy as np

UNSUPPORTED = (
    "use_dynamic_shifting",
    "use_karras_sigmas",
    "use_exponential_sigmas",
    "use_beta_sigmas",
    "invert_sigmas",
    "stochastic_sampling",
    "shift_terminal",
)


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0, **config):
        enabled = [name for name in UNSUPPORTED if config.get(name)]
        if enabled:
            raise NotImplementedError(f"FlowMatchEulerDiscreteScheduler options not ported: {enabled}")
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        # __init__: float32 sigmas of the full training schedule; only their ends are used later.
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        sigmas = timesteps / np.float32(num_train_timesteps)
        sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).astype(np.float32)
        self.sigma_max, self.sigma_min = float(sigmas[0]), float(sigmas[-1])
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(sigmas * np.float32(num_train_timesteps))
        self._sigmas_np = sigmas
        self._step_index = None

    @classmethod
    def from_config(cls, config: dict) -> "FlowMatchEulerDiscreteScheduler":
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        if config.get("time_shift_type", "exponential") not in ("exponential", "linear"):
            raise ValueError("time_shift_type must be 'exponential' or 'linear'")
        return cls(**config)

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(self, num_inference_steps: int) -> None:
        timesteps = np.linspace(
            self.sigma_max * self.num_train_timesteps, self.sigma_min * self.num_train_timesteps, num_inference_steps
        )
        sigmas = timesteps / self.num_train_timesteps
        sigmas = (self.shift * sigmas / (1 + (self.shift - 1) * sigmas)).astype(np.float32)
        timesteps = sigmas * np.float32(self.num_train_timesteps)
        self._sigmas_np = np.concatenate([sigmas, np.zeros(1, np.float32)])
        self.sigmas = mx.array(self._sigmas_np)
        self.timesteps = mx.array(timesteps)
        self._timesteps_np = timesteps
        self._step_index = None

    def _index_for(self, timestep) -> int:
        value = float(np.asarray(timestep).reshape(-1)[0])
        matches = np.nonzero(self._timesteps_np == np.float32(value))[0]
        if len(matches) == 0:
            raise ValueError(f"timestep {value} is not in the schedule")
        return int(matches[1] if len(matches) > 1 else matches[0])

    def step(self, model_output: mx.array, timestep, sample: mx.array) -> mx.array:
        if self._step_index is None:
            self._step_index = self._index_for(timestep)
        sigma, sigma_next = self._sigmas_np[self._step_index], self._sigmas_np[self._step_index + 1]
        dt = mx.array(sigma_next - sigma, dtype=mx.float32).astype(model_output.dtype)
        update = (model_output.astype(mx.float32) * dt.astype(mx.float32)).astype(model_output.dtype)
        prev = sample.astype(mx.float32) + update.astype(mx.float32)
        self._step_index += 1
        return prev.astype(model_output.dtype)
