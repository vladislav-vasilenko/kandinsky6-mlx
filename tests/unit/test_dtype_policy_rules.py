"""Spec 14: the two float32 rules for the DiT, on real tensor names."""

import pytest

from kandinsky6_mlx.utils.dtype_policy import keeps_fp32


@pytest.mark.parametrize(
    ("name", "substring", "diffusers"),
    [
        ("out_layer.modulation.out_layer.weight", True, True),
        ("audio_out_layer.modulation.out_layer.bias", True, True),
        ("visual_transformer_blocks.0.video_dec_block.visual_modulation.out_layer.weight", True, False),
        ("visual_transformer_blocks.0.va_modulation.out_layer.weight", True, False),
        ("audio_text_transformer_blocks.0.text_modulation.out_layer.bias", True, False),
        ("audio_time_embeddings.timestep_embedder.linear_1.weight", True, False),
        ("visual_embeddings.in_layer.weight", False, False),
    ],
)
def test_rules(name, substring, diffusers):
    assert keeps_fp32("transformer", name) is diffusers
    assert keeps_fp32("transformer", name, "substring") is substring
    assert keeps_fp32("transformer", name, "diffusers") is diffusers


def test_other_components_and_unknown_rule():
    assert keeps_fp32("vae", "decoder.modulation.weight", "diffusers") is False
    with pytest.raises(ValueError, match="unknown float32 rule"):
        keeps_fp32("transformer", "x", "other")
