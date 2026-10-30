"""BigVGAN-v2 vocoder (`MMAudioVocoder`) in MLX.

Mirrors `diffusers/models/autoencoders/mmaudio_vocoder.py` (diffusers 0.42.1) with the same parameter paths. Activations
are channels-last `(B, L, C)` inside the model; `__call__` takes the reference layout `(B, num_mels, frames)` and
returns `(B, 1, samples)`. Convolution weights are in the MLX layout `[out, kernel, in]` written by the converter; the
low-pass filter buffers keep the reference shape `(1, 1, kernel)`.
"""

import math

import mlx.core as mx
import numpy as np
from mlx import nn


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """Kaiser-windowed sinc low-pass filter `(1, 1, kernel_size)` with unit sum.

    Only the initial value: the converted checkpoint stores the filters as buffers, which `load_weights` replaces.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta)
    time = np.arange(-half_size, half_size) + 0.5 if even else np.arange(kernel_size) - half_size
    taps = 2 * cutoff * window * np.sinc(2 * cutoff * time)
    return mx.array((taps / taps.sum()).astype(np.float32).reshape(1, 1, kernel_size))


class MMAudioSnakeBeta(nn.Module):
    """`x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) * x) ** 2` with per-channel log-scale `alpha` and `beta`."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha, beta = mx.exp(self.alpha), mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.square(mx.sin(x * alpha))


def _depthwise(filter: mx.array, channels: int) -> mx.array:
    """Reference `(1, 1, K)` filter -> MLX depthwise weight `(C, K, 1)`."""
    return mx.broadcast_to(filter.reshape(1, -1, 1), (channels, filter.shape[-1], 1))


class MMAudioLowPassFilter1d(nn.Module):
    """Replicate padding, then a depthwise convolution with the low-pass filter at `stride`."""

    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode="edge")
        return mx.conv1d(x, _depthwise(self.filter, channels), stride=self.stride, groups=channels)


class MMAudioUpSample1d(nn.Module):
    """Replicate padding, a depthwise transposed convolution with the low-pass filter, scaling and cropping."""

    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.filter = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        channels = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        x = self.ratio * mx.conv_transpose1d(x, _depthwise(self.filter, channels), stride=self.stride, groups=channels)
        return x[:, self.pad_left : x.shape[1] - self.pad_right]


class MMAudioDownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MMAudioLowPassFilter1d(0.5 / ratio, 0.6 / ratio, stride=ratio, kernel_size=kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class MMAudioActivation1d(nn.Module):
    """Anti-aliased activation: 2x upsample, SnakeBeta, 2x downsample."""

    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = MMAudioSnakeBeta(channels)
        self.upsample = MMAudioUpSample1d(ratio, kernel_size)
        self.downsample = MMAudioDownSample1d(ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class MMAudioAMPBlock(nn.Module):
    """Anti-aliased multi-periodicity block: dilated convolutions each followed by a dilation-1 convolution."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilations
        ]
        self.convs2 = [nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2) for _ in dilations]
        self.activations = [MMAudioActivation1d(channels) for _ in range(2 * len(dilations))]

    def __call__(self, x: mx.array) -> mx.array:
        activations_1, activations_2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, activations_1, activations_2):
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class MMAudioVocoder(nn.Module):
    """Mel spectrogram `(B, num_mels, frames)` -> waveform `(B, 1, frames * prod(upsample_rates))` in `[-1, 1]`."""

    def __init__(
        self,
        num_mels: int = 128,
        upsample_initial_channel: int = 1536,
        upsample_rates: tuple[int, ...] = (8, 4, 2, 2, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (16, 8, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("`upsample_rates` and `upsample_kernel_sizes` must have the same length")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError("`resblock_kernel_sizes` and `resblock_dilation_sizes` must have the same length")
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, padding=3)
        ups, resblocks = [], []
        channels = upsample_initial_channel
        for rate, kernel_size in zip(upsample_rates, upsample_kernel_sizes):
            ups.append(
                [nn.ConvTranspose1d(channels, channels // 2, kernel_size, rate, padding=(kernel_size - rate) // 2)]
            )
            channels //= 2
            for block_kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                resblocks.append(MMAudioAMPBlock(channels, block_kernel_size, tuple(dilations)))
        self.ups = ups
        self.resblocks = resblocks
        self.activation_post = MMAudioActivation1d(channels)
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3, bias=False)

    @classmethod
    def from_config(cls, config: dict) -> "MMAudioVocoder":
        names = (
            "num_mels",
            "upsample_initial_channel",
            "upsample_rates",
            "upsample_kernel_sizes",
            "resblock_kernel_sizes",
            "resblock_dilation_sizes",
        )
        return cls(**{key: config[key] for key in names if key in config})

    def __call__(self, mel: mx.array) -> mx.array:
        hidden_states = self.conv_pre(mel.transpose(0, 2, 1))
        for stage, up in enumerate(self.ups):
            hidden_states = up[0](hidden_states)
            blocks = self.resblocks[stage * self.num_kernels : (stage + 1) * self.num_kernels]
            # Python's sum starts from 0, as in the reference.
            hidden_states = sum(block(hidden_states) for block in blocks) / self.num_kernels
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return mx.clip(hidden_states, -1.0, 1.0).transpose(0, 2, 1)
