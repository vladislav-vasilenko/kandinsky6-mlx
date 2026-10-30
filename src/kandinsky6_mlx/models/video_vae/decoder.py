"""Decoder of the HunyuanVideo causal 3D VAE (`AutoencoderKLHunyuanVideo.decode`) in MLX.

Mirrors `diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py` (diffusers 0.42.1) with the same parameter
paths (`decoder.*`, `post_quant_conv`); the encoder is not ported. Activations are channels-last `(B, T, H, W, C)`
inside the model; `decode` takes and returns the reference layout `(B, C, T, H, W)`. Convolution weights are in the
MLX layout `[out, kT, kH, kW, in]` written by the converter.

Like the reference, `decode` splits more than `tile_sample_min_num_frames // 4` latent frames into overlapping
temporal tiles and blends them (`use_framewise_decoding`, on by default); spatial tiling is spec 10.
"""

import math

import mlx.core as mx
from mlx import nn


class HunyuanVideoCausalConv3d(nn.Module):
    """Replicate padding (`kernel - 1` frames in front only, `kernel // 2` on each spatial side), then a 3D conv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        k = self.kernel_size
        if k > 1:
            pad = [(0, 0), (k - 1, 0), (k // 2, k // 2), (k // 2, k // 2), (0, 0)]
            x = mx.pad(x, pad, mode="edge")
        return self.conv(x)


class HunyuanVideoUpsampleCausal3D(nn.Module):
    """Nearest upsampling: the first frame spatially only, the other frames by `upsample_factor` (T, H, W)."""

    def __init__(self, in_channels: int, out_channels: int | None = None, upsample_factor=(2, 2, 2)):
        super().__init__()
        self.upsample_factor = tuple(upsample_factor)
        self.conv = HunyuanVideoCausalConv3d(in_channels, out_channels or in_channels, 3)

    def __call__(self, x: mx.array) -> mx.array:
        f_t, f_h, f_w = self.upsample_factor

        def spatial(y: mx.array) -> mx.array:
            return mx.repeat(mx.repeat(y, f_h, axis=2), f_w, axis=3)

        first = spatial(x[:, :1])
        if x.shape[1] > 1:
            x = mx.concatenate([first, mx.repeat(spatial(x[:, 1:]), f_t, axis=1)], axis=1)
        else:
            x = first
        return self.conv(x)


def _group_norm(groups: int, channels: int, eps: float) -> nn.GroupNorm:
    """`torch.nn.GroupNorm`: contiguous channel groups, statistics over the group's channels and all positions."""
    return nn.GroupNorm(groups, channels, eps=eps, affine=True, pytorch_compatible=True)


class HunyuanVideoResnetBlockCausal3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = _group_norm(groups, in_channels, eps)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, 3)
        self.norm2 = _group_norm(groups, out_channels, eps)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, 3)
        if in_channels != out_channels:
            self.conv_shortcut = HunyuanVideoCausalConv3d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        hidden_states = self.conv1(nn.silu(self.norm1(x)))
        hidden_states = self.conv2(nn.silu(self.norm2(hidden_states)))
        residual = self.conv_shortcut(x) if "conv_shortcut" in self else x
        return hidden_states + residual


def causal_frame_mask(num_frames: int, tokens_per_frame: int) -> mx.array:
    """`prepare_causal_attention_mask` as a boolean mask: a token sees every token of its own and earlier frames."""
    frame = mx.repeat(mx.arange(num_frames), tokens_per_frame)
    return frame[None, :] <= frame[:, None]


class _Attention(nn.Module):
    """diffusers `Attention` as built by the mid block: GroupNorm, one head over all channels, residual."""

    def __init__(self, channels: int, groups: int, eps: float):
        super().__init__()
        self.group_norm = _group_norm(groups, channels, eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        batch, frames, height, width, channels = x.shape
        tokens = x.reshape(batch, -1, channels)
        normed = self.group_norm(tokens)
        query, key, value = (proj(normed)[:, None] for proj in (self.to_q, self.to_k, self.to_v))
        mask = causal_frame_mask(frames, height * width)
        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=channels**-0.5, mask=mask)
        out = self.to_out[0](out[:, 0]) + tokens
        return out.reshape(batch, frames, height, width, channels)


class HunyuanVideoMidBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int, eps: float = 1e-6, add_attention: bool = True):
        super().__init__()
        self.resnets = [HunyuanVideoResnetBlockCausal3D(channels, channels, groups, eps) for _ in range(2)]
        self.attentions = [_Attention(channels, groups, eps)] if add_attention else [None]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            if attention is not None:
                x = attention(x)
            x = resnet(x)
        return x


class HunyuanVideoUpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, groups, eps=1e-6, upsample_factor=None):
        super().__init__()
        self.resnets = [
            HunyuanVideoResnetBlockCausal3D(in_channels if i == 0 else out_channels, out_channels, groups, eps)
            for i in range(num_layers)
        ]
        if upsample_factor is not None:
            self.upsamplers = [HunyuanVideoUpsampleCausal3D(out_channels, out_channels, upsample_factor)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if "upsamplers" in self:
            x = self.upsamplers[0](x)
        return x


class HunyuanVideoDecoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        if time_compression_ratio != 4:
            raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")
        self.conv_in = HunyuanVideoCausalConv3d(in_channels, block_out_channels[-1], 3)
        self.mid_block = HunyuanVideoMidBlock3D(
            block_out_channels[-1], norm_num_groups, add_attention=mid_block_add_attention
        )
        reversed_channels = list(reversed(block_out_channels))
        num_spatial = int(math.log2(spatial_compression_ratio))
        num_time = int(math.log2(time_compression_ratio))
        up_blocks, output_channel = [], reversed_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel, output_channel = output_channel, reversed_channels[i]
            is_final = i == len(block_out_channels) - 1
            spatial = i < num_spatial
            temporal = i >= len(block_out_channels) - 1 - num_time and not is_final
            factor = ((2,) if temporal else (1,)) + ((2, 2) if spatial else (1, 1))
            up_blocks.append(
                HunyuanVideoUpBlock3D(
                    prev_output_channel,
                    output_channel,
                    layers_per_block + 1,
                    norm_num_groups,
                    upsample_factor=factor if (spatial or temporal) else None,
                )
            )
        self.up_blocks = up_blocks
        self.conv_norm_out = _group_norm(norm_num_groups, block_out_channels[0], 1e-6)
        self.conv_out = HunyuanVideoCausalConv3d(block_out_channels[0], out_channels, 3)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mid_block(self.conv_in(x))
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class AutoencoderKLHunyuanVideo(nn.Module):
    """`AutoencoderKLHunyuanVideo` with `decode` only."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
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
        self.temporal_compression_ratio = temporal_compression_ratio
        self.decoder = HunyuanVideoDecoder3D(
            latent_channels,
            out_channels,
            tuple(block_out_channels),
            layers_per_block,
            norm_num_groups,
            mid_block_add_attention,
            temporal_compression_ratio,
            spatial_compression_ratio,
        )
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)
        # Reference defaults (`AutoencoderKLHunyuanVideo.__init__`).
        self.use_framewise_decoding = True
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_num_frames = 12

    @classmethod
    def from_config(cls, config: dict) -> "AutoencoderKLHunyuanVideo":
        names = (
            "in_channels",
            "out_channels",
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
    def decoder_weights(weights: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
        """The tensors `decode` needs out of a full converted `vae` file (drops the encoder and `quant_conv`)."""
        return [(name, value) for name, value in weights.items() if name.startswith(("decoder.", "post_quant_conv."))]

    def _decode_tile(self, z: mx.array) -> mx.array:
        return self.decoder(self.post_quant_conv(z))

    def _temporal_tiled_decode(self, z: mx.array) -> mx.array:
        num_frames = z.shape[1]
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
        tile_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        stride_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        row = []
        for i in range(0, num_frames, stride_frames):
            decoded = self._decode_tile(z[:, i : i + tile_frames + 1])
            if i > 0:
                decoded = decoded[:, 1:]
            mx.eval(decoded)
            row.append(decoded)
        result = []
        for i in range(len(row)):
            if i > 0:
                # The reference blends in place, so tile i is blended against the already blended tile i - 1.
                row[i] = self._blend_t(row[i - 1], row[i], blend_frames)
                result.append(row[i][:, : self.tile_sample_stride_num_frames])
            else:
                result.append(row[i][:, : self.tile_sample_stride_num_frames + 1])
        return mx.concatenate(result, axis=1)[:, :num_sample_frames]

    @staticmethod
    def _blend_t(a: mx.array, b: mx.array, blend_extent: int) -> mx.array:
        blend_extent = min(a.shape[1], b.shape[1], blend_extent)
        frames = [
            a[:, -blend_extent + x] * (1 - x / blend_extent) + b[:, x] * (x / blend_extent) for x in range(blend_extent)
        ]
        return mx.concatenate([mx.stack(frames, axis=1), b[:, blend_extent:]], axis=1) if frames else b

    def decode(self, z: mx.array) -> mx.array:
        """Latents already divided by `scaling_factor`, `(B, C, T, H, W)` -> frames `(B, 3, 4(T-1)+1, 8H, 8W)`."""
        z = z.transpose(0, 2, 3, 4, 1)
        tile_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        if self.use_framewise_decoding and z.shape[1] > tile_frames:
            out = self._temporal_tiled_decode(z)
        else:
            out = self._decode_tile(z)
        return out.transpose(0, 4, 1, 2, 3)
