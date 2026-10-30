"""The Kandinsky 6 DiT, `Kandinsky6Transformer3DModel`, in MLX.

Mirrors `diffusers/models/transformers/transformer_kandinsky6.py` (diffusers 0.42.1): same constructor arguments,
module names and parameter paths, so a converted `model.safetensors` loads without renaming.
"""

import inspect
import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_map_with_path

from kandinsky6_mlx.models.dit.blocks import (
    Kandinsky6FusedTransformerDecoderBlock,
    Kandinsky6OutLayer,
    Kandinsky6OutLayerAudio,
    Kandinsky6TransformerEncoderBlock,
)
from kandinsky6_mlx.models.dit.primitives import (
    Kandinsky6RoPE1D,
    Kandinsky6RoPE3D,
    Kandinsky6TextEmbeddings,
    Kandinsky6TimeEmbeddings,
    Kandinsky6VisualEmbeddings,
)
from kandinsky6_mlx.utils.dtype_policy import keeps_fp32

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class Kandinsky6Transformer3DModel(nn.Module):
    """Video and audio denoiser joined by fused cross-modal blocks, each conditioned on its own text branch."""

    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        in_text_dim: int = 3584,
        in_text_dim2: int = 768,
        time_dim: int = 1024,
        patch_size: tuple = (1, 2, 2),
        model_dim: int = 4096,
        ff_dim: int = 16384,
        num_text_blocks: int = 4,
        num_visual_blocks: int = 60,
        axes_dims: tuple = (32, 48, 48),
        visual_cond: bool = True,
        in_audio_dim: int = 20,
        out_audio_dim: int = 20,
        model_dim_a: int | None = None,
        time_dim_a: int | None = None,
        ff_dim_a: int | None = None,
        axes_dims_a: tuple | None = None,
        audio_freqs_scaling: float = 1.0,
        scale_factor: tuple | list[float] = (1.0, 2.0, 2.0),
        text_token_padding: bool = False,
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        visual_token_type_num_embeddings: int = 0,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.visual_cond = visual_cond
        self.in_visual_dim = in_visual_dim
        self.in_audio_dim = in_audio_dim
        self.text_token_padding = text_token_padding
        self.scale_factor = tuple(float(value) for value in scale_factor)
        self.visual_token_type_num_embeddings = int(visual_token_type_num_embeddings or 0)
        head_dim = sum(axes_dims)
        model_dim_a = model_dim_a or model_dim
        time_dim_a = time_dim_a or time_dim
        ff_dim_a = ff_dim_a or ff_dim
        axes_dims_a = axes_dims_a or axes_dims
        head_dim_a = sum(axes_dims_a)

        vis_in_dim = (2 * in_visual_dim + 1) if visual_cond else in_visual_dim
        self.visual_embeddings = Kandinsky6VisualEmbeddings(vis_in_dim, model_dim, self.patch_size)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(self.visual_token_type_num_embeddings, model_dim)
        self.visual_rope_embeddings = Kandinsky6RoPE3D(tuple(axes_dims))
        self.out_layer = Kandinsky6OutLayer(model_dim, time_dim, out_visual_dim, self.patch_size)

        self.audio_embeddings = Kandinsky6TextEmbeddings(in_audio_dim, model_dim_a)
        self.audio_rope_embeddings = Kandinsky6RoPE1D(head_dim_a, freqs_scaling=audio_freqs_scaling)
        self.audio_out_layer = Kandinsky6OutLayerAudio(model_dim_a, time_dim_a, out_audio_dim)
        for prefix, md, td, fd, hd in (
            ("video", model_dim, time_dim, ff_dim, head_dim),
            ("audio", model_dim_a, time_dim_a, ff_dim_a, head_dim_a),
        ):
            setattr(self, f"{prefix}_time_embeddings", Kandinsky6TimeEmbeddings(md, td))
            setattr(self, f"{prefix}_text_embeddings", Kandinsky6TextEmbeddings(in_text_dim, md))
            setattr(self, f"{prefix}_pooled_text_embeddings", Kandinsky6TextEmbeddings(in_text_dim2, td))
            setattr(self, f"{prefix}_text_rope_embeddings", Kandinsky6RoPE1D(hd))
            setattr(
                self,
                f"{prefix}_text_transformer_blocks",
                [Kandinsky6TransformerEncoderBlock(md, td, fd, hd) for _ in range(num_text_blocks)],
            )
        self.visual_transformer_blocks = [
            Kandinsky6FusedTransformerDecoderBlock(
                model_dim,
                time_dim,
                ff_dim,
                head_dim,
                model_dim_a,
                time_dim_a,
                ff_dim_a,
                head_dim_a,
                ca_rope,
                cross_gates,
                fix_modulation,
            )
            for _ in range(num_visual_blocks)
        ]

    @classmethod
    def from_config(cls, config: dict) -> "Kandinsky6Transformer3DModel":
        """Build from a `config.json` dict; keys the constructor does not take (e.g. `is_multimodal`) are ignored."""
        accepted = inspect.signature(cls.__init__).parameters
        return cls(**{key: value for key, value in config.items() if key in accepted})

    @classmethod
    def from_converted(cls, path: str | Path, dtype: mx.Dtype = mx.bfloat16) -> "Kandinsky6Transformer3DModel":
        """Load a converted transformer directory (`config.json` + `model.safetensors`).

        Parameters are cast to `dtype`, except time embeddings and every modulation, which run in float32 as with
        diffusers' `_keep_in_fp32_modules` (the Lite checkpoint stores 268 of them as BF16). Arrays are memory-mapped
        and only read when evaluated.
        """
        path = Path(path)
        model = cls.from_config(json.loads((path / CONFIG_NAME).read_text()))
        model.load_weights(str(path / WEIGHTS_NAME), strict=True)
        model.apply_dtype_policy(dtype)
        mx.eval(model.parameters())
        return model

    def apply_dtype_policy(self, dtype: mx.Dtype) -> None:
        """Cast floating parameters to `dtype`, keeping `keeps_fp32` ones in float32."""

        def cast(name: str, value: mx.array) -> mx.array:
            if not mx.issubdtype(value.dtype, mx.floating):
                return value
            return value.astype(mx.float32 if keeps_fp32("transformer", name) else dtype)

        self.update(tree_map_with_path(cast, self.parameters()))

    @property
    def dtype(self) -> mx.Dtype:
        """Compute dtype of the model: the dtype of its input projection."""
        return self.visual_embeddings.in_layer.weight.dtype

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        pooled_projections: mx.array,
        timestep: mx.array,
        audio_hidden_states: mx.array | None = None,
        visual_rope_pos: tuple[mx.array, mx.array, mx.array] | None = None,
        encoder_attention_mask: mx.array | None = None,
        visual_token_type_ids: mx.array | None = None,
    ) -> tuple[mx.array] | tuple[mx.array, mx.array]:
        """Predict velocities; returns `(video,)` or `(video, audio)` like the reference with `return_dict=False`.

        `hidden_states` is `(B, T, H, W, C)` with `C = 2 * in_visual_dim + 1` when `visual_cond`, `timestep` is on the
        `[0, num_train_timesteps]` scale. Floating inputs are cast to the model dtype.
        """
        if visual_token_type_ids is not None and self.visual_token_type_num_embeddings == 0:
            raise ValueError("`visual_token_type_ids` requires `visual_token_type_num_embeddings > 0`")
        dtype = self.dtype
        hidden_states = hidden_states.astype(dtype)
        encoder_hidden_states = encoder_hidden_states.astype(dtype)
        pooled_projections = pooled_projections.astype(dtype)

        # 1. Rotary embeddings for the two text branches
        text_pos = mx.arange(encoder_hidden_states.shape[1])
        video_text_rope = self.video_text_rope_embeddings(text_pos)
        audio_text_rope = self.audio_text_rope_embeddings(text_pos)

        # 2. Encode the text tokens through the video and audio text branches
        video_text_embed = self.video_text_embeddings(encoder_hidden_states)
        video_temb = self.video_time_embeddings(timestep) + self.video_pooled_text_embeddings(pooled_projections)
        for block in self.video_text_transformer_blocks:
            video_text_embed = block(video_text_embed, video_temb, video_text_rope, encoder_attention_mask)

        audio_text_embed = self.audio_text_embeddings(encoder_hidden_states)
        audio_temb = self.audio_time_embeddings(timestep) + self.audio_pooled_text_embeddings(pooled_projections)
        for block in self.audio_text_transformer_blocks:
            audio_text_embed = block(audio_text_embed, audio_temb, audio_text_rope, encoder_attention_mask)

        # 3. Patchify the video latents and build their rotary embeddings
        visual_embed = self.visual_embeddings(hidden_states)
        if visual_token_type_ids is not None:
            token_types = self.visual_token_type_embeddings(visual_token_type_ids)
            visual_embed = visual_embed + token_types[:, :, None, None, :]
        visual_shape = tuple(visual_embed.shape[1:4])
        if visual_rope_pos is None:
            visual_rope_pos = tuple(mx.arange(size) for size in visual_shape)
        visual_rope = self.visual_rope_embeddings(visual_rope_pos, self.scale_factor)
        visual_embed = visual_embed.reshape(visual_embed.shape[0], -1, visual_embed.shape[-1])
        visual_rope = visual_rope.reshape(-1, *visual_rope.shape[3:])

        # 4. Embed the audio latents and build their rotary embeddings
        audio_embed = audio_rope = None
        if audio_hidden_states is not None:
            audio_embed = self.audio_embeddings(audio_hidden_states.astype(dtype))
            audio_rope = self.audio_rope_embeddings(mx.arange(audio_hidden_states.shape[1]))

        # 5. Run the fused video/audio transformer blocks (a block skips the audio path when it is absent)
        for block in self.visual_transformer_blocks:
            visual_embed, audio_embed = block(
                visual_embed,
                audio_embed,
                video_text_embed,
                audio_text_embed,
                (video_temb, audio_temb),
                visual_rope,
                audio_rope,
                encoder_attention_mask,
            )

        # 6. Project back to the latent space
        visual_embed = visual_embed.reshape(-1, *visual_shape, visual_embed.shape[-1])
        video_out = self.out_layer(visual_embed, video_temb)
        if audio_embed is None:
            return (video_out,)
        return video_out, self.audio_out_layer(audio_embed, audio_temb)


def parameter_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """Flat parameter names and shapes, in the reference `state_dict` naming."""
    return {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
