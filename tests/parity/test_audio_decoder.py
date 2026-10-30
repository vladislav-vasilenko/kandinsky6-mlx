"""Spec 07: MLX MMAudio VAE decoder against diffusers 0.42.1."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.models.audio_vae.decoder import MMAudioVAE, normalize
from kandinsky6_mlx.utils.compare import compare
from tests.parity.audio_cases import audio_vae_results, tiny_audio_vae, to_mlx_layout

CONFIGS = Path(__file__).resolve().parents[2] / "fixtures" / "configs"
RESULTS = audio_vae_results()
CLASSES = {"MMAudioMPConv1d", "MMAudioResnetBlock1D", "MMAudioAttnBlock1D", "MMAudioUpsample1D", "MMAudioDecoder1D"}


def test_every_decoder_class_is_covered():
    assert {r["class"] for r in RESULTS} == CLASSES


@pytest.mark.parametrize("result", RESULTS, ids=[r["module"] for r in RESULTS])
def test_recorded_call_matches_reference(result):
    assert result["rel_max"] <= 1e-4, result


def test_decode_matches_reference():
    model, tensors, _ = tiny_audio_vae()
    assert compare(model.decode(tensors["inputs.decode.z"]), tensors["outputs.decode.sample"])["rel_max"] <= 1e-4


def test_normalize_matches_reference_for_attention_layout():
    """`normalize(qkv, dim=2)` over (B, heads, D, 3, L) in the reference is the D axis of (B, L, heads, D, 3) here."""
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.autoencoder_mmaudio import normalize as reference

    x = np.random.default_rng(0).standard_normal((2, 2, 5, 3, 7)).astype(np.float32)
    ref = reference(torch.from_numpy(x), dim=2).numpy()
    got = np.array(normalize(mx.array(x.transpose(0, 4, 1, 2, 3)), axes=(3,))).transpose(0, 2, 3, 4, 1)
    assert compare(got, ref)["rel_max"] <= 1e-6


def _reference_config() -> dict:
    return json.loads((CONFIGS / "lite-distill-audio_vae.json").read_text())


def test_released_config_uses_reference_defaults():
    """The released config predates the diffusers arguments (`mode`, `downsample_factor`); both sides use defaults."""
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE as Reference

    with torch.device("meta"):
        reference = Reference.from_config(_reference_config())
    model = MMAudioVAE.from_config(_reference_config())
    assert (model.scaling_factor, model.sample_rate, model.latent_hop_length) == (0.417, 44100, 1024)
    assert reference.latent_hop_length == model.latent_hop_length
    expected = {}
    for name, value in reference.state_dict().items():
        if name.startswith(("vae.decoder.", "vae.data_")):
            rule = conv_rule("audio_vae", name)
            shape = tuple(value.shape)
            expected[name] = tuple(shape[a] for a in rule.axes()) if rule else shape
    got = {name: tuple(v.shape) for name, v in tree_flatten(model.parameters())}
    assert got == expected


@pytest.mark.parametrize("latent_length", [45, 218], ids=["smoke", "baseline"])
def test_released_structure_lengths_and_values(latent_length):
    """Released structure (40 latent channels, levels (1, 2, 4), 3 blocks per level) at a small width."""
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE as Reference

    from reference.make_tiny_fixtures import build

    config = {"mel_bins": 16, "hidden_channels": 8}
    reference = build(Reference, config)
    model = MMAudioVAE.from_config(config)
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    model.load_weights(MMAudioVAE.decoder_weights(to_mlx_layout("audio_vae", weights)), strict=True)
    z = np.random.default_rng(0).standard_normal((1, 40, latent_length)).astype(np.float32)
    with torch.no_grad():
        ref = reference.decode(torch.from_numpy(z), return_dict=False)[0].numpy()
    got = model.decode(mx.array(z))
    assert got.shape == ref.shape == (1, 16, 2 * latent_length)
    assert compare(got, ref)["rel_max"] <= 1e-4
