"""CLIP-L text tower in MLX for the pooled prompt embedding of Kandinsky 6 (`text_encoder_2`).

Mirrors `transformers.CLIPTextModel` (5.22.1) as Kandinsky 6 calls it: token + learned position embeddings, pre-norm
transformer layers with a causal attention mask and `quick_gelu`, a final LayerNorm, and `pooler_output` = the final
state at the end-of-text position. With `eos_token_id == 2` (the released config) transformers takes that position as
the argmax of the token ids, i.e. the first occurrence of the largest id (49407, `<|endoftext|>`), which is also what
the padding uses. Parameter names follow the checkpoint (`embeddings.*`, `encoder.layers.N.*`, `final_layer_norm.*`).
Right padding cannot reach the pooled position under the causal mask, so the padding mask is not needed.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn


def quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        b, n, d = x.shape

        def heads(t):
            return t.reshape(b, n, self.heads, d // self.heads).transpose(0, 2, 1, 3)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, n, d))


class CLIPMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(quick_gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = CLIPAttention(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = CLIPMLP(dim, hidden)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class CLIPEmbeddings(nn.Module):
    def __init__(self, vocab: int, dim: int, positions: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dim)
        self.position_embedding = nn.Embedding(positions, dim)


class CLIPEncoder(nn.Module):
    def __init__(self, layers: int, dim: int, heads: int, hidden: int, eps: float):
        super().__init__()
        self.layers = [CLIPEncoderLayer(dim, heads, hidden, eps) for _ in range(layers)]


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 77,
        layer_norm_eps: float = 1e-5,
        hidden_act: str = "quick_gelu",
        eos_token_id: int = 2,
    ):
        super().__init__()
        if hidden_act != "quick_gelu":
            raise NotImplementedError(f"hidden_act {hidden_act!r}")
        if eos_token_id != 2:
            # transformers then pools at the first `eos_token_id` instead of the argmax; not needed for CLIP-L.
            raise NotImplementedError("only the legacy argmax pooling (eos_token_id == 2) is ported")
        self.embeddings = CLIPEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = CLIPEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, layer_norm_eps
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    @classmethod
    def from_pretrained(cls, directory: str | Path, dtype: mx.Dtype = mx.float32) -> "CLIPTextModel":
        """A transformers `CLIPTextModel` directory (`config.json`, `model.safetensors`), e.g. `text_encoder_2/`."""
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        names = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
            "layer_norm_eps",
            "hidden_act",
            "eos_token_id",
        )
        model = cls(**{k: config[k] for k in names if k in config})
        weights = mx.load(str(directory / "model.safetensors"))
        weights = {k.removeprefix("text_model."): v.astype(dtype) for k, v in weights.items()}
        weights.pop("embeddings.position_ids", None)
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        return model

    def __call__(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """`(B, L)` token ids -> (`last_hidden_state` `(B, L, D)`, `pooler_output` `(B, D)`)."""
        n = input_ids.shape[1]
        x = self.embeddings.token_embedding(input_ids) + self.embeddings.position_embedding(mx.arange(n))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(n).astype(x.dtype)
        for layer in self.encoder.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        # torch.argmax returns the first maximal index; so does NumPy (MLX's tie order is not documented).
        eos = mx.array(np.asarray(input_ids).argmax(axis=-1))
        pooled = x[mx.arange(x.shape[0]), eos]
        return x, pooled
