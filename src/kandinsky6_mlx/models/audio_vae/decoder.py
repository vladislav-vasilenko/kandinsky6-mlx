"""Decoder of the MMAudio mel-spectrogram VAE in MLX.

Mirrors `diffusers/models/autoencoders/autoencoder_mmaudio.py` (diffusers 0.42.1) with the same parameter paths
(`vae.decoder.*`, `vae.data_mean`, `vae.data_std`); the encoder and the mel front end are not ported. Activations are
channels-last `(B, L, C)` inside the model; `decode` takes and returns the reference layout `(B, C, L)`. Convolution
weights are in the MLX layout `[out, kernel, in]` written by the converter.
"""

import math

import mlx.core as mx
from mlx import nn

ACTIVATION_CLIP = 256.0


def normalize(x: mx.array, axes: tuple[int, ...], eps: float = 1e-4) -> mx.array:
    """Rescale `x` to unit RMS over `axes`: `x / (eps + ||x|| * sqrt(norm.numel() / x.numel()))`."""
    norm = mx.sqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=axes, keepdims=True))
    norm = eps + norm * math.sqrt(norm.size / x.size)
    return x / norm.astype(x.dtype)


def mp_silu(x: mx.array) -> mx.array:
    """SiLU rescaled so that a unit-variance input stays unit-variance."""
    return nn.silu(x) / 0.596


def mp_sum(a: mx.array, b: mx.array, t: float = 0.5) -> mx.array:
    """`a.lerp(b, t)` rescaled to unit variance; torch's lerp is `a + t * (b - a)` for `t < 0.5`."""
    lerp = a + t * (b - a) if t < 0.5 else b - (1 - t) * (b - a)
    return lerp / math.sqrt((1 - t) ** 2 + t**2)


class MMAudioMPConv1d(nn.Module):
    """Magnitude-preserving 1D convolution (no bias, padding `kernel // 2`) with an optional per-call gain."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size // 2
        scale = 1 / math.sqrt(in_channels * kernel_size)
        self.weight = mx.random.uniform(-scale, scale, (out_channels, kernel_size, in_channels))

    def __call__(self, x: mx.array, gain: float | mx.array = 1.0) -> mx.array:
        return mx.conv1d(x, (self.weight * gain).astype(x.dtype), padding=self.padding)


class MMAudioResnetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = MMAudioMPConv1d(in_channels, out_channels, kernel_size)
        self.conv2 = MMAudioMPConv1d(out_channels, out_channels, kernel_size)
        if in_channels != out_channels:
            self.nin_shortcut = MMAudioMPConv1d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = normalize(x, axes=(-1,))
        hidden_states = self.conv1(mp_silu(x))
        hidden_states = self.conv2(mp_silu(hidden_states))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return mp_sum(x, hidden_states, t=0.3)


class MMAudioAttnBlock1D(nn.Module):
    """Self-attention over time with q/k/v normalized per head, residual through `mp_sum`."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = MMAudioMPConv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = MMAudioMPConv1d(channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, channels = x.shape
        # The reference reshapes (B, 3C, L) to (B, heads, D, 3, L): q, k and v are interleaved along the channels.
        qkv = self.qkv(x).reshape(batch, length, self.num_heads, -1, 3)
        qkv = normalize(qkv, axes=(3,))
        query, key, value = (qkv[..., i].transpose(0, 2, 1, 3) for i in range(3))  # (B, heads, L, D)
        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=query.shape[-1] ** -0.5)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, channels)
        return mp_sum(x, self.proj_out(out), t=0.3)


class MMAudioUpsample1D(nn.Module):
    """Nearest-exact 2x upsampling (each frame repeated twice) followed by an MP convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = MMAudioMPConv1d(channels, channels, kernel_size=3)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(mx.repeat(x, 2, axis=1))


class _Mid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = MMAudioResnetBlock1D(channels, channels)
        self.attn_1 = MMAudioAttnBlock1D(channels)
        self.block_2 = MMAudioResnetBlock1D(channels, channels)


class _Up(nn.Module):
    def __init__(self, blocks: list, upsample: MMAudioUpsample1D | None):
        super().__init__()
        self.block = blocks
        if upsample is not None:
            self.upsample = upsample


class MMAudioDecoder1D(nn.Module):
    """Latents to standardized log-mel frames; one 2x temporal upsample after the second-to-last level."""

    def __init__(
        self,
        mel_bins: int,
        latent_channels: int,
        hidden_channels: int,
        channel_multipliers: tuple[int, ...],
        layers_per_block: int,
    ):
        super().__init__()
        block_in = hidden_channels * channel_multipliers[-1]
        self.conv_in = MMAudioMPConv1d(latent_channels, block_in, kernel_size=3)
        self.mid = _Mid(block_in)
        up = []
        for level in reversed(range(len(channel_multipliers))):
            block_out = hidden_channels * channel_multipliers[level]
            blocks = []
            for _ in range(layers_per_block + 1):
                blocks.append(MMAudioResnetBlock1D(block_in, block_out))
                block_in = block_out
            up.insert(0, _Up(blocks, MMAudioUpsample1D(block_in) if level == 1 else None))
        self.up = up
        self.conv_out = MMAudioMPConv1d(block_in, mel_bins, kernel_size=3)
        self.learnable_gain = mx.zeros(())

    def __call__(self, z: mx.array) -> mx.array:
        hidden_states = self.conv_in(z)
        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = mx.clip(self.mid.block_2(hidden_states), -ACTIVATION_CLIP, ACTIVATION_CLIP)
        for level in reversed(range(len(self.up))):
            up = self.up[level]
            for block in up.block:
                hidden_states = mx.clip(block(hidden_states), -ACTIVATION_CLIP, ACTIVATION_CLIP)
            if "upsample" in up:
                hidden_states = up.upsample(hidden_states)
        return self.conv_out(mp_silu(hidden_states), gain=self.learnable_gain + 1)


class MMAudioAutoencoder(nn.Module):
    """Decoder half of the autoencoder plus the per-bin statistics `data_mean`, `data_std` of shape `(1, mel, 1)`."""

    def __init__(self, mel_bins, latent_channels, hidden_channels, channel_multipliers, layers_per_block):
        super().__init__()
        self.data_mean = mx.zeros((1, mel_bins, 1))
        self.data_std = mx.ones((1, mel_bins, 1))
        self.decoder = MMAudioDecoder1D(
            mel_bins, latent_channels, hidden_channels, channel_multipliers, layers_per_block
        )


class MMAudioVAE(nn.Module):
    """`MMAudioVAE` with `decode` only: latents `(B, latent_channels, N)` to a mel spectrogram `(B, mel_bins, 2N)`."""

    def __init__(
        self,
        mel_bins: int = 128,
        latent_channels: int = 40,
        hidden_channels: int = 512,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        layers_per_block: int = 2,
        sample_rate: int = 44_100,
        n_fft: int = 2048,
        hop_length: int = 512,
        scaling_factor: float = 0.417,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.scaling_factor = scaling_factor
        # The encoder downsamples the mel frames once by 2.
        self.latent_hop_length = hop_length * 2
        self.vae = MMAudioAutoencoder(
            mel_bins, latent_channels, hidden_channels, tuple(channel_multipliers), layers_per_block
        )

    @classmethod
    def from_config(cls, config: dict) -> "MMAudioVAE":
        names = (
            "mel_bins",
            "latent_channels",
            "hidden_channels",
            "channel_multipliers",
            "layers_per_block",
            "sample_rate",
            "n_fft",
            "hop_length",
            "scaling_factor",
        )
        return cls(**{key: config[key] for key in names if key in config})

    @staticmethod
    def decoder_weights(weights: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
        """The tensors `decode` needs out of a full converted `audio_vae` file (drops the encoder and mel front end)."""
        keep = ("vae.decoder.", "vae.data_mean", "vae.data_std")
        return [(name, value) for name, value in weights.items() if name.startswith(keep)]

    def decode(self, z: mx.array) -> mx.array:
        """Latents already divided by `scaling_factor`, `(B, C, N)` -> mel `(B, mel_bins, 2N)`."""
        mel = self.vae.decoder(z.transpose(0, 2, 1)).transpose(0, 2, 1)
        return mel * self.vae.data_std + self.vae.data_mean
