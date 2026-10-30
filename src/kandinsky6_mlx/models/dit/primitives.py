"""Building blocks of the Kandinsky 6 DiT in MLX.

Mirrors `diffusers/models/transformers/transformer_kandinsky6.py` (diffusers 0.42.1): module names and parameter
paths are the same, so converted weights load without renaming. Tensors use the reference layouts
(`(B, L, H, D)` inside attention, `(B, T, H, W, C)` for video latents).
"""

import math

import mlx.core as mx
import numpy as np
from mlx import nn


def _exp_correctly_rounded(argument: np.ndarray) -> mx.array:
    """`exp` of a float32 argument, computed in float64 and rounded once to float32.

    Frequencies are constants, so they are computed as accurately as possible instead of copying one device's float32
    `exp` (torch CPU, torch CUDA and MLX each differ in the last bit for some inputs).
    """
    return mx.array(np.exp(argument.astype(np.float64)).astype(np.float32))


def get_freqs(dim: int, max_period: float = 10000.0) -> mx.array:
    """Inverse rotary frequencies in float32; the argument is formed in float32 as in the reference."""
    argument = (np.float32(-math.log(max_period)) * np.arange(dim, dtype=np.float32)) / np.float32(dim)
    return _exp_correctly_rounded(argument)


def _broadcast_condition(x: mx.array, condition: mx.array) -> mx.array:
    """Reshape a `(B, C)` condition so it broadcasts over the token dimensions of `x`."""
    if x.ndim > 2 and condition.ndim == 2:
        return condition.reshape(condition.shape[0], *([1] * (x.ndim - 2)), condition.shape[-1])
    return condition


def apply_scale_shift(normed: mx.array, x: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    """AdaLN scale/shift of an already-normalized tensor, computed in float32 and cast back to `x.dtype`."""
    scale, shift = _broadcast_condition(x, scale), _broadcast_condition(x, shift)
    out = normed * (scale.astype(mx.float32) + 1.0) + shift.astype(mx.float32)
    return out.astype(x.dtype)


def apply_gate_sum(x: mx.array, out: mx.array, gate: mx.array) -> mx.array:
    """Gated residual `x + gate * out` in float32, cast back to `x.dtype`."""
    gate = _broadcast_condition(x, gate)
    result = x.astype(mx.float32) + gate.astype(mx.float32) * out.astype(mx.float32)
    return result.astype(x.dtype)


def apply_rotary(x: mx.array, rope: mx.array) -> mx.array:
    """Rotate pairs of the last dimension of `x` with 2x2 matrices from `rope`, in float32."""
    pairs = x.reshape(*x.shape[:-1], -1, 1, 2).astype(mx.float32)
    return (rope.astype(mx.float32) * pairs).sum(axis=-1).reshape(x.shape).astype(x.dtype)


def _rotation_matrices(angles: mx.array) -> mx.array:
    """`(..., n)` angles -> `(..., n, 2, 2)` rotation matrices `[[cos, -sin], [sin, cos]]`."""
    cos, sin = mx.cos(angles), mx.sin(angles)
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*angles.shape, 2, 2)


class Kandinsky6RoPE1D(nn.Module):
    """1D rotary embedding for text and audio tokens. Returns `(L, 1, dim // 2, 2, 2)`."""

    def __init__(self, dim: int, max_pos: int = 2048, max_period: float = 10000.0, freqs_scaling: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        self._freq = get_freqs(dim // 2, max_period) * freqs_scaling
        self.freeze()

    def __call__(self, pos: mx.array) -> mx.array:
        angles = pos.astype(mx.float32)[:, None] * self._freq[None, :]
        return _rotation_matrices(angles)[:, None]


class Kandinsky6RoPE3D(nn.Module):
    """3D rotary embedding for video tokens. Returns `(T, H, W, 1, sum(axes_dims) // 2, 2, 2)`."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.max_pos = tuple(max_pos)
        self._freqs = [get_freqs(d // 2, max_period) for d in self.axes_dims]
        self.freeze()

    def __call__(
        self,
        pos: tuple[mx.array, mx.array, mx.array],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> mx.array:
        num_frames, height, width = (int(axis_pos.shape[0]) for axis_pos in pos)
        angles_t, angles_h, angles_w = (
            (axis_pos.astype(mx.float32)[:, None] * freq[None, :]) / factor
            for axis_pos, freq, factor in zip(pos, self._freqs, scale_factor)
        )
        shape = (num_frames, height, width)
        angles = mx.concatenate(
            [
                mx.broadcast_to(angles_t[:, None, None, :], (*shape, angles_t.shape[-1])),
                mx.broadcast_to(angles_h[None, :, None, :], (*shape, angles_h.shape[-1])),
                mx.broadcast_to(angles_w[None, None, :, :], (*shape, angles_w.shape[-1])),
            ],
            axis=-1,
        )
        return _rotation_matrices(angles)[:, :, :, None]


class RMSNorm(nn.Module):
    """`torch.nn.RMSNorm(dims)` with `eps=None`.

    PyTorch computes it in float32 for float32/bfloat16/float16 inputs: input and weight are upcast, the epsilon is
    `finfo(float32).eps` (not the input dtype's), and the result is cast back to the input dtype. Normalizing a bf16
    input in bf16 with the bf16 epsilon (0.0078) would change outputs by tens of percent.
    """

    def __init__(self, dims: int, eps: float | None = None):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        eps = self.eps if self.eps is not None else mx.finfo(mx.float32).eps
        out = mx.fast.rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32), eps)
        return out.astype(x.dtype)


class Kandinsky6Attention(nn.Module):
    """Multi-head attention with per-head RMSNorm on queries and keys and optional rotary embeddings."""

    def __init__(self, num_channels: int, head_dim: int, kv_dim: int | None = None):
        super().__init__()
        if num_channels % head_dim:
            raise ValueError("num_channels must be divisible by head_dim")
        kv_dim = kv_dim or num_channels
        self.num_heads = num_channels // head_dim
        self.to_query = nn.Linear(num_channels, num_channels)
        self.to_key = nn.Linear(kv_dim, num_channels)
        self.to_value = nn.Linear(kv_dim, num_channels)
        self.query_norm = RMSNorm(head_dim)
        self.key_norm = RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attn_mask: mx.array | None = None,
        rotary_emb: mx.array | None = None,
        rope_q: mx.array | None = None,
        rope_kv: mx.array | None = None,
    ) -> mx.array:
        rotary_emb = rope_q if rope_q is not None else rotary_emb
        rotary_emb_kv = rope_kv if rope_kv is not None else rotary_emb
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        query = self.to_query(hidden_states)
        key = self.to_key(context)
        value = self.to_value(context)
        query = query.reshape(*query.shape[:-1], self.num_heads, -1)
        key = key.reshape(*key.shape[:-1], self.num_heads, -1)
        value = value.reshape(*value.shape[:-1], self.num_heads, -1)
        query = self.query_norm(query)
        key = self.key_norm(key)
        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb)
        if rotary_emb_kv is not None:
            key = apply_rotary(key, rotary_emb_kv)

        # Reference layout (B, L, H, D); MLX attention takes (B, H, L, D).
        query, key, value = (t.transpose(0, 2, 1, 3) for t in (query, key, value))
        mask = None
        if attn_mask is not None:
            # A (B, L_kv) padding mask hides padded keys for every query, as diffusers' native backend does.
            mask = attn_mask.astype(mx.bool_)[:, None, None, :] if attn_mask.ndim == 2 else attn_mask
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=query.shape[-1] ** -0.5, mask=mask)
        output = output.transpose(0, 2, 1, 3)
        return self.out_layer(output.reshape(*output.shape[:-2], -1))


class _Dropout(nn.Module):
    """Placeholder at `net.1` of the reference `FeedForward` (dropout is inactive at inference)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class GELU(nn.Module):
    """Linear projection followed by exact (erf) GELU, `diffusers.models.activations.GELU`."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.proj(x))


class FeedForward(nn.Module):
    """`diffusers.models.attention.FeedForward(activation_fn="gelu")`: parameters `net.0.proj` and `net.2`."""

    def __init__(self, dim: int, inner_dim: int, dim_out: int | None = None, bias: bool = True):
        super().__init__()
        self.net = [GELU(dim, inner_dim, bias=bias), _Dropout(), nn.Linear(inner_dim, dim_out or dim, bias=bias)]

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.net:
            x = module(x)
        return x


def timestep_embedding(timesteps: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """`diffusers.models.embeddings.get_timestep_embedding` with `flip_sin_to_cos=True`, shift 0, scale 1."""
    half = dim // 2
    exponent = (np.float32(-math.log(max_period)) * np.arange(half, dtype=np.float32)) / np.float32(half)
    emb = timesteps.astype(mx.float32)[:, None] * _exp_correctly_rounded(exponent)[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class TimestepEmbedding(nn.Module):
    """`diffusers.models.embeddings.TimestepEmbedding(act_fn="silu")`."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class Timesteps(nn.Module):
    """Parameter-free sinusoidal projection at `time_proj`."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def __call__(self, timesteps: mx.array) -> mx.array:
        return timestep_embedding(timesteps, self.num_channels)


class Kandinsky6TimeEmbeddings(nn.Module):
    """Sinusoidal timestep embedding followed by a SiLU MLP; runs in the dtype of its weights (float32)."""

    def __init__(self, model_dim: int, time_dim: int):
        super().__init__()
        if model_dim % 2:
            raise ValueError("model_dim must be even")
        self.time_proj = Timesteps(model_dim)
        self.timestep_embedder = TimestepEmbedding(model_dim, time_dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        embed = self.time_proj(timestep).astype(self.timestep_embedder.linear_1.weight.dtype)
        return self.timestep_embedder(embed)


class Kandinsky6TextEmbeddings(nn.Module):
    """Linear projection followed by LayerNorm with affine parameters."""

    def __init__(self, text_dim: int, model_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, affine=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.in_layer(x))


class Kandinsky6VisualEmbeddings(nn.Module):
    """Patchify `(B, T, H, W, C)` latents into `(B, T', H', W', prod(patch) * C)` and project them."""

    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.in_layer = nn.Linear(math.prod(self.patch_size) * visual_dim, model_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.in_layer(patchify(x, self.patch_size))


def patchify(x: mx.array, patch_size: tuple[int, int, int]) -> mx.array:
    """Element order of the reference: `view(...).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)`."""
    batch, duration, height, width, channels = x.shape
    p_t, p_h, p_w = patch_size
    x = x.reshape(batch, duration // p_t, p_t, height // p_h, p_h, width // p_w, p_w, channels)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(batch, duration // p_t, height // p_h, width // p_w, -1)


class Kandinsky6Modulation(nn.Module):
    """AdaLN modulation: SiLU then a linear layer producing `num_params * model_dim` values, in its weight dtype."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(x.astype(self.out_layer.weight.dtype)))
