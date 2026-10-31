"""Qwen2.5-VL language model in MLX for Kandinsky 6 text embeddings, via `mlx-vlm` (extra `text`).

Kandinsky 6 uses `hidden_states[-1]` of Qwen2.5-VL-7B-Instruct, i.e. the last decoder state after the final RMSNorm.
`mlx_vlm.models.qwen2_5_vl.language.Qwen2Model.__call__` returns exactly `self.norm(h)`; `LanguageModel` would add
`lm_head`, so the inner model is used. Text-only M-RoPE positions are `arange` on all three axes, which is what the
model builds when `position_ids` is not given. The vision tower and `lm_head` are not loaded.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from mlx_vlm.models.qwen2_5_vl.language import Qwen2Model


class QwenTextEncoder:
    def __init__(self, model: Qwen2Model):
        self.model = model

    @classmethod
    def from_files(
        cls, config: str | Path, weights: str | Path | list[str | Path], dtype: mx.Dtype = mx.bfloat16
    ) -> "QwenTextEncoder":
        """`config`: a Qwen2.5-VL `config.json`; `weights`: safetensors file(s) with HF names (`model.layers.*`), e.g.
        the sharded `text_encoder/` of the checkpoint or a single-file copy such as ComfyUI's
        `qwen_2.5_vl_7b.safetensors`. Only `model.*` (embeddings, decoder layers, final norm) is kept."""
        text_config = TextConfig.from_dict(json.loads(Path(config).read_text()))
        model = Qwen2Model(text_config)
        files = weights if isinstance(weights, list) else [weights]
        tensors = []
        for file in files:
            for name, value in mx.load(str(file)).items():
                if name.startswith("model."):
                    tensors.append((name[len("model.") :], value.astype(dtype)))
        model.load_weights(tensors, strict=True)
        mx.eval(model.parameters())
        return cls(model)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """`(B, L)` token ids -> `(B, L, hidden)` last hidden states, causal attention, no padding inside."""
        return self.model(input_ids)
