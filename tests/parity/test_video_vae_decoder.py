"""Spec 08: MLX HunyuanVideo causal 3D VAE decoder against diffusers 0.42.1."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.models.video_vae.decoder import (
    AutoencoderKLHunyuanVideo,
    HunyuanVideoCausalConv3d,
    HunyuanVideoUpsampleCausal3D,
    causal_frame_mask,
)
from kandinsky6_mlx.utils.compare import compare
from tests.parity.audio_cases import to_mlx_layout
from tests.parity.video_cases import tiny_video_vae, video_vae_results

CONFIGS = Path(__file__).resolve().parents[2] / "fixtures" / "configs"
RESULTS = video_vae_results()
CLASSES = {
    "Conv3d",
    "HunyuanVideoCausalConv3d",
    "HunyuanVideoResnetBlockCausal3D",
    "HunyuanVideoMidBlock3D",
    "HunyuanVideoUpsampleCausal3D",
    "HunyuanVideoUpBlock3D",
    "HunyuanVideoDecoder3D",
}


def test_every_decoder_class_is_covered():
    assert {r["class"] for r in RESULTS} == CLASSES


@pytest.mark.parametrize("result", RESULTS, ids=[r["module"] for r in RESULTS])
def test_recorded_call_matches_reference(result):
    assert result["rel_max"] <= 1e-4, result


def test_decode_matches_reference():
    model, tensors, _ = tiny_video_vae()
    assert compare(model.decode(tensors["inputs.decode.z"]), tensors["outputs.decode.sample"])["rel_max"] <= 1e-4


def test_causal_padding_on_distinct_integers():
    """Replicate padding: two copies of frame 0 in front, nothing behind, one replicated row/column per side."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    x = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 1, 3, 4, 5)
    ref = F.pad(torch.from_numpy(x), (1, 1, 1, 1, 2, 0), mode="replicate").numpy()
    conv = HunyuanVideoCausalConv3d(1, 1, 3)
    # An identity-on-the-first-tap kernel exposes the padded input element for element.
    weight = np.zeros((1, 3, 3, 3, 1), np.float32)
    weight[0, 0, 0, 0, 0] = 1
    conv.conv.weight, conv.conv.bias = mx.array(weight), mx.zeros((1,))
    got = np.array(conv(mx.array(x.transpose(0, 2, 3, 4, 1))))[..., 0]
    np.testing.assert_array_equal(got[0], ref[0, 0, : got.shape[1], : got.shape[2], : got.shape[3]])
    assert got.shape[1:] == x.shape[2:]


@pytest.mark.parametrize("frames", [1, 2, 7])
@pytest.mark.parametrize("factor", [(1, 2, 2), (2, 2, 2)])
def test_upsample_frame_counts_and_values(frames, factor):
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoUpsampleCausal3D as Reference

    torch.manual_seed(0)
    reference = Reference(4, 4, upsample_factor=factor).eval()
    module = HunyuanVideoUpsampleCausal3D(4, 4, factor)
    weights = {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    module.load_weights(list(to_mlx_layout("vae", weights).items()), strict=True)
    x = np.random.default_rng(0).standard_normal((1, 4, frames, 3, 5)).astype(np.float32)
    with torch.no_grad():
        ref = reference(torch.from_numpy(x)).numpy()
    got = np.array(module(mx.array(x.transpose(0, 2, 3, 4, 1)))).transpose(0, 4, 1, 2, 3)
    assert got.shape[2] == ref.shape[2] == (1 + factor[0] * (frames - 1))
    assert compare(got, ref)["rel_max"] <= 1e-5


@pytest.mark.parametrize(("frames", "tokens"), [(1, 4), (3, 2), (5, 6)])
def test_causal_frame_mask_matches_reference(frames, tokens):
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import prepare_causal_attention_mask

    ref = prepare_causal_attention_mask(frames, tokens, torch.float32, "cpu").numpy() == 0
    np.testing.assert_array_equal(np.array(causal_frame_mask(frames, tokens)), ref)


def test_released_config_builds_reference_parameters():
    torch = pytest.importorskip("torch")
    from diffusers import AutoencoderKLHunyuanVideo as Reference

    config = json.loads((CONFIGS / "lite-distill-vae.json").read_text())
    with torch.device("meta"):
        reference = Reference.from_config(config)
    expected = {}
    for name, value in reference.state_dict().items():
        if name.startswith(("decoder.", "post_quant_conv.")):
            rule = conv_rule("vae", name)
            shape = tuple(value.shape)
            expected[name] = tuple(shape[a] for a in rule.axes()) if rule else shape
    got = {name: tuple(v.shape) for name, v in tree_flatten(AutoencoderKLHunyuanVideo.from_config(config).parameters())}
    assert got == expected


def _small_released_structure():
    """Released structure (4 up blocks, 3 resnets each, x8 spatial, x4 temporal, mid attention) at a small width."""
    import torch
    from diffusers import AutoencoderKLHunyuanVideo as Reference

    from reference.make_tiny_fixtures import build

    config = {"block_out_channels": (8, 8, 16, 16), "norm_num_groups": 4, "layers_per_block": 2}
    reference = build(Reference, config)
    model = AutoencoderKLHunyuanVideo.from_config(config)
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    model.load_weights(AutoencoderKLHunyuanVideo.decoder_weights(to_mlx_layout("vae", weights)), strict=True)
    return torch, reference, model


@pytest.mark.parametrize("frames", [1, 2, 7], ids=["T1", "T2", "T7-framewise"])
def test_released_structure_decode(frames):
    """T = 7 (Smoke) takes the reference's default temporal tiling: tiles of 5 latent frames, stride 3, blend 4."""
    pytest.importorskip("torch")
    torch, reference, model = _small_released_structure()
    z = np.random.default_rng(0).standard_normal((1, 16, frames, 4, 6)).astype(np.float32)
    with torch.no_grad():
        ref = reference.decode(torch.from_numpy(z), return_dict=False)[0].numpy()
    got = model.decode(mx.array(z))
    assert got.shape == ref.shape == (1, 3, 4 * (frames - 1) + 1, 32, 48)
    assert compare(got, ref)["rel_max"] <= 1e-4


def test_framewise_off_matches_reference_without_tiling():
    pytest.importorskip("torch")
    torch, reference, model = _small_released_structure()
    reference.use_framewise_decoding = model.use_framewise_decoding = False
    z = np.random.default_rng(1).standard_normal((1, 16, 7, 4, 6)).astype(np.float32)
    with torch.no_grad():
        ref = reference.decode(torch.from_numpy(z), return_dict=False)[0].numpy()
    assert compare(model.decode(mx.array(z)), ref)["rel_max"] <= 1e-4


def test_chunked_causal_conv_equals_one_pass(monkeypatch):
    """Convolving in chunks of output frames (to bound mx.conv3d's temporary buffer) gives the same result."""
    rng = np.random.default_rng(0)
    conv = HunyuanVideoCausalConv3d(8, 8, 3)
    x = mx.array(rng.standard_normal((1, 9, 5, 6, 8)).astype(np.float32))
    whole = conv(x)
    monkeypatch.setattr(HunyuanVideoCausalConv3d, "chunk_bytes", 1)  # one output frame per chunk
    chunked = conv(x)
    assert chunked.shape == whole.shape == (1, 9, 5, 6, 8)
    assert compare(chunked, whole)["rel_max"] <= 1e-6
