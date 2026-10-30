"""Transformer blocks and output heads of the Kandinsky 6 DiT in MLX.

Mirrors `diffusers/models/transformers/transformer_kandinsky6.py` (diffusers 0.42.1) with the same module names and
parameter paths. Norms run on float32 copies of the stream and every AdaLN affine and gated residual is computed in
float32 and cast back to the stream dtype, as in the reference.
"""

import math

import mlx.core as mx
from mlx import nn

from kandinsky6_mlx.models.dit.primitives import (
    FeedForward,
    Kandinsky6Attention,
    Kandinsky6Modulation,
    apply_gate_sum,
    apply_scale_shift,
)


def _layer_norm(dims: int) -> nn.LayerNorm:
    """`torch.nn.LayerNorm(dims, elementwise_affine=False)`."""
    return nn.LayerNorm(dims, eps=1e-5, affine=False)


def _normed(norm: nn.LayerNorm, x: mx.array) -> mx.array:
    """The reference calls norms on `x.float()`."""
    return norm(x.astype(mx.float32))


def unpatchify(x: mx.array, patch_size: tuple[int, int, int]) -> mx.array:
    """Element order of the reference video head.

    `view(B, T, H, W, -1, p_t, p_h, p_w).permute(0, 1, 5, 2, 6, 3, 7, 4)` then merge each axis with its patch axis.
    The channel axis is the slowest within a patch, unlike `patchify` where it is the fastest.
    """
    batch, duration, height, width = x.shape[:4]
    p_t, p_h, p_w = patch_size
    x = x.reshape(batch, duration, height, width, -1, p_t, p_h, p_w)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(batch, duration * p_t, height * p_h, width * p_w, -1)


class Kandinsky6OutLayer(nn.Module):
    """Projects visual hidden states `(B, T, H, W, C)` back to latent patches and unpatchifies them."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = _layer_norm(model_dim)
        self.out_layer = nn.Linear(model_dim, math.prod(self.patch_size) * visual_dim)

    def __call__(self, visual_embed: mx.array, time_embed: mx.array) -> mx.array:
        shift, scale = mx.split(self.modulation(time_embed), 2, axis=-1)
        x = apply_scale_shift(_normed(self.norm, visual_embed), visual_embed, scale, shift)
        return unpatchify(self.out_layer(x), self.patch_size)


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects audio hidden states back to audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = _layer_norm(model_dim)
        self.out_layer = nn.Linear(model_dim, audio_dim)

    def __call__(self, audio_embed: mx.array, time_embed: mx.array) -> mx.array:
        shift, scale = mx.split(self.modulation(time_embed), 2, axis=-1)
        x = apply_scale_shift(_normed(self.norm, audio_embed), audio_embed, scale, shift)
        # The reference normalizes a second time after the affine (in the stream dtype); the weights expect it.
        x = self.norm(x)
        return self.out_layer(x)


class Kandinsky6TransformerEncoderBlock(nn.Module):
    """Text self-attention and feed-forward with AdaLN modulation."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int):
        super().__init__()
        self.text_modulation = Kandinsky6Modulation(time_dim, model_dim, 6)
        self.attn_norm = _layer_norm(model_dim)
        self.attn = Kandinsky6Attention(model_dim, head_dim)
        self.feed_forward_norm = _layer_norm(model_dim)
        self.feed_forward = FeedForward(model_dim, ff_dim, bias=False)

    def __call__(
        self, x: mx.array, time_embed: mx.array, rope: mx.array, attn_mask: mx.array | None = None
    ) -> mx.array:
        sa_params, ff_params = mx.split(self.text_modulation(time_embed), 2, axis=-1)
        shift, scale, gate = mx.split(sa_params, 3, axis=-1)
        normed = apply_scale_shift(_normed(self.attn_norm, x), x, scale, shift)
        x = apply_gate_sum(x, self.attn(normed, rotary_emb=rope, attn_mask=attn_mask), gate)
        shift, scale, gate = mx.split(ff_params, 3, axis=-1)
        normed = apply_scale_shift(_normed(self.feed_forward_norm, x), x, scale, shift)
        return apply_gate_sum(x, self.feed_forward(normed), gate)


class Kandinsky6TransformerDecoderBlock(nn.Module):
    """Container of the per-modality submodules of the fused block; it has no forward of its own."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int):
        super().__init__()
        self.visual_modulation = Kandinsky6Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = _layer_norm(model_dim)
        self.self_attention = Kandinsky6Attention(model_dim, head_dim)
        self.cross_attention_norm = _layer_norm(model_dim)
        self.cross_attention = Kandinsky6Attention(model_dim, head_dim, kv_dim=model_dim)
        self.feed_forward_norm = _layer_norm(model_dim)
        self.feed_forward = FeedForward(model_dim, ff_dim, bias=False)

    def self_and_text_attention(
        self, x: mx.array, time_embed: mx.array, rope: mx.array | None, text: mx.array, attn_mask: mx.array | None
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Self-attention with its gated residual, then text cross-attention **without** its residual.

        Returns `(x, pre_ca, out_t, gate_t, ff_params)`: the stream after self-attention, the modulated normalized
        stream used as the cross-attention query (and as cross-modal keys/values), the text cross-attention output,
        its gate, and the feed-forward modulation parameters.
        """
        sa_params, ca_params, ff_params = mx.split(self.visual_modulation(time_embed), 3, axis=-1)
        shift, scale, gate = mx.split(sa_params, 3, axis=-1)
        normed = apply_scale_shift(_normed(self.self_attention_norm, x), x, scale, shift)
        x = apply_gate_sum(x, self.self_attention(normed, rotary_emb=rope), gate)
        shift, scale, gate_t = mx.split(ca_params, 3, axis=-1)
        pre_ca = apply_scale_shift(_normed(self.cross_attention_norm, x), x, scale, shift)
        out_t = self.cross_attention(pre_ca, encoder_hidden_states=text, attn_mask=attn_mask)
        return x, pre_ca, out_t, gate_t, ff_params

    def feed_forward_residual(self, x: mx.array, ff_params: mx.array) -> mx.array:
        shift, scale, gate = mx.split(ff_params, 3, axis=-1)
        normed = apply_scale_shift(_normed(self.feed_forward_norm, x), x, scale, shift)
        return apply_gate_sum(x, self.feed_forward(normed), gate)


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Video and audio decoder blocks joined by gated cross-modal attention."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
    ):
        super().__init__()
        self.video_dec_block = Kandinsky6TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim)
        self.audio_dec_block = Kandinsky6TransformerDecoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a)
        self.va_cross_attention = Kandinsky6Attention(model_dim, head_dim, kv_dim=model_dim_a)
        self.av_cross_attention = Kandinsky6Attention(model_dim_a, head_dim_a, kv_dim=model_dim)
        self.va_modulation = Kandinsky6Modulation(
            time_dim, model_dim if not cross_gates else model_dim * 2 + model_dim_a, 1 if cross_gates else 3
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a, model_dim_a if not cross_gates else model_dim_a * 2 + model_dim, 1 if cross_gates else 3
        )
        self.va_normalization = _layer_norm(model_dim)
        self.av_normalization = _layer_norm(model_dim_a)
        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def _cross_modal_params(self, t_v: mx.array, t_a: mx.array) -> tuple[list[mx.array], list[mx.array]]:
        """`(shift, scale, gate)` for the video-from-audio and audio-from-video attentions, as computed."""
        # Without fix_modulation the reference conditions each direction on the other modality's time embedding.
        va_params = self.va_modulation(t_v if self.fix_modulation else t_a)
        av_params = self.av_modulation(t_a if self.fix_modulation else t_v)
        if self.cross_gates:
            va = mx.split(va_params, [self.model_dim, 2 * self.model_dim], axis=-1)
            av = mx.split(av_params, [self.model_dim_a, 2 * self.model_dim_a], axis=-1)
        else:
            va = mx.split(va_params, 3, axis=-1)
            av = mx.split(av_params, 3, axis=-1)
        return va, av

    def _cross_modal_gates(self, va_gate: mx.array, av_gate: mx.array) -> tuple[mx.array, mx.array]:
        """Gates of the video and audio cross-modal residuals; `cross_gates` swaps them."""
        return (av_gate, va_gate) if self.cross_gates else (va_gate, av_gate)

    def __call__(
        self,
        vis: mx.array | None,
        aud: mx.array | None,
        text_v: mx.array,
        text_a: mx.array,
        time_embed: tuple[mx.array, mx.array],
        vis_rope: mx.array | None,
        aud_rope: mx.array | None,
        attn_mask: mx.array | None = None,
    ) -> tuple[mx.array | None, mx.array | None]:
        t_v, t_a = time_embed
        if vis is not None:
            vis, vis_pre_ca, vis_out_t, gate_v, ff_params_v = self.video_dec_block.self_and_text_attention(
                vis, t_v, vis_rope, text_v, attn_mask
            )

        if aud is not None:
            aud, aud_pre_ca, aud_out_t, gate_a, ff_params_a = self.audio_dec_block.self_and_text_attention(
                aud, t_a, aud_rope, text_a, attn_mask
            )
            aud = apply_gate_sum(aud, aud_out_t, gate_a)

            if vis is not None:
                (va_shift, va_scale, va_gate), (av_shift, av_scale, av_gate) = self._cross_modal_params(t_v, t_a)
                vis = apply_gate_sum(vis, vis_out_t, gate_v)
                vis_for_va = apply_scale_shift(_normed(self.va_normalization, vis), vis, va_scale, va_shift)
                aud_for_av = apply_scale_shift(_normed(self.av_normalization, aud), aud, av_scale, av_shift)
                rope_v = vis_rope if self.ca_rope else None
                rope_a = aud_rope if self.ca_rope else None
                # Keys and values are the pre-cross-attention states of the other modality, not the updated streams.
                vis_from_aud = self.va_cross_attention(
                    vis_for_va, encoder_hidden_states=aud_pre_ca, rope_q=rope_v, rope_kv=rope_a
                )
                aud_from_vis = self.av_cross_attention(
                    aud_for_av, encoder_hidden_states=vis_pre_ca, rope_q=rope_a, rope_kv=rope_v
                )
                vis_gate, aud_gate = self._cross_modal_gates(va_gate, av_gate)
                vis = apply_gate_sum(vis, vis_from_aud, vis_gate)
                aud = apply_gate_sum(aud, aud_from_vis, aud_gate)
        elif vis is not None:
            vis = apply_gate_sum(vis, vis_out_t, gate_v)

        if vis is not None:
            vis = self.video_dec_block.feed_forward_residual(vis, ff_params_v)
        if aud is not None:
            aud = self.audio_dec_block.feed_forward_residual(aud, ff_params_a)
        return vis, aud
