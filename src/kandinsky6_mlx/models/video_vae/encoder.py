"""Encoder of the HunyuanVideo causal 3D VAE (`AutoencoderKLHunyuanVideo.encode`) in MLX, for image-to-video.

Mirrors `diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py` (diffusers 0.42.1):
`HunyuanVideoEncoder3D` (conv_in, down blocks of causal resnets with strided causal downsampling convs, the mid block
with causal attention, GroupNorm, SiLU, conv_out to 2 x latent channels), then `quant_conv` and
`DiagonalGaussianDistribution` (mean and log-variance clamped to [-30, 20]). Parameter paths are the checkpoint's
(`encoder.*`, `quant_conv.*`); convolution weights are in the MLX layout written by the converter. Activations are
channels-last `(B, T, H, W, C)` inside; `encode` takes and returns the reference layout `(B, C, T, H, W)`.

Only plain encoding is ported: the pipeline encodes one frame, below `tile_sample_min_num_frames`, and does not enable
tiling unless asked (`tiled_encode` is not ported).
"""

import math

import mlx.core as mx
from mlx import nn

from kandinsky6_mlx.models.video_vae.decoder import (
    HunyuanVideoCausalConv3d,
    HunyuanVideoMidBlock3D,
    HunyuanVideoResnetBlockCausal3D,
    _group_norm,
)


class HunyuanVideoDownsampleCausal3D(nn.Module):
    def __init__(self, channels: int, stride: tuple[int, int, int]):
        super().__init__()
        self.conv = HunyuanVideoCausalConv3d(channels, channels, 3, stride)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class HunyuanVideoDownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, groups, eps=1e-6, downsample_stride=None):
        super().__init__()
        self.resnets = [
            HunyuanVideoResnetBlockCausal3D(in_channels if i == 0 else out_channels, out_channels, groups, eps)
            for i in range(num_layers)
        ]
        if downsample_stride is not None:
            self.downsamplers = [HunyuanVideoDownsampleCausal3D(out_channels, downsample_stride)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if "downsamplers" in self:
            x = self.downsamplers[0](x)
        return x


class HunyuanVideoEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        if temporal_compression_ratio != 4:
            raise ValueError(f"Unsupported time_compression_ratio: {temporal_compression_ratio}")
        self.conv_in = HunyuanVideoCausalConv3d(in_channels, block_out_channels[0], 3)
        num_spatial = int(math.log2(spatial_compression_ratio))
        num_time = int(math.log2(temporal_compression_ratio))
        blocks, output_channel = [], block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel, output_channel = output_channel, block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            spatial = i < num_spatial
            temporal = i >= len(block_out_channels) - 1 - num_time and not is_final
            stride = ((2,) if temporal else (1,)) + ((2, 2) if spatial else (1, 1))
            blocks.append(
                HunyuanVideoDownBlock3D(
                    input_channel,
                    output_channel,
                    layers_per_block,
                    norm_num_groups,
                    downsample_stride=stride if (spatial or temporal) else None,
                )
            )
        self.down_blocks = blocks
        self.mid_block = HunyuanVideoMidBlock3D(
            block_out_channels[-1], norm_num_groups, add_attention=mid_block_add_attention
        )
        self.conv_norm_out = _group_norm(norm_num_groups, block_out_channels[-1], 1e-6)
        self.conv_out = HunyuanVideoCausalConv3d(block_out_channels[-1], 2 * out_channels, 3)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class HunyuanVideoVAEEncoder(nn.Module):
    """`encoder` + `quant_conv` of `AutoencoderKLHunyuanVideo`, returning the latent distribution's parameters."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.476986,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.spatial_compression_ratio = spatial_compression_ratio
        self.encoder = HunyuanVideoEncoder3D(
            in_channels,
            latent_channels,
            tuple(block_out_channels),
            layers_per_block,
            norm_num_groups,
            mid_block_add_attention,
            temporal_compression_ratio,
            spatial_compression_ratio,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)

    @classmethod
    def from_config(cls, config: dict) -> "HunyuanVideoVAEEncoder":
        names = (
            "in_channels",
            "latent_channels",
            "block_out_channels",
            "layers_per_block",
            "norm_num_groups",
            "scaling_factor",
            "spatial_compression_ratio",
            "temporal_compression_ratio",
            "mid_block_add_attention",
        )
        return cls(**{key: config[key] for key in names if key in config})

    @staticmethod
    def encoder_weights(weights: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
        """The tensors `encode` needs out of a full converted `vae` file (drops the decoder and `post_quant_conv`)."""
        return [(name, value) for name, value in weights.items() if name.startswith(("encoder.", "quant_conv."))]

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Pixels in [-1, 1], `(B, 3, T, H, W)` -> (mean, logvar), each `(B, 16, (T-1)/4+1, H/8, W/8)`."""
        moments = self.quant_conv(self.encoder(x.transpose(0, 2, 3, 4, 1))).transpose(0, 4, 1, 2, 3)
        mean, logvar = mx.split(moments, 2, axis=1)
        return mean, mx.clip(logvar, -30.0, 20.0)

    @staticmethod
    def sample(mean: mx.array, logvar: mx.array, noise: mx.array) -> mx.array:
        """`DiagonalGaussianDistribution.sample` with explicit standard-normal `noise` of the mean's shape."""
        return mean + mx.exp(0.5 * logvar) * noise
