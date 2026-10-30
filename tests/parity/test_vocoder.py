"""Spec 07: MLX BigVGAN vocoder against diffusers 0.42.1."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.models.vocoder.bigvgan import MMAudioVocoder, kaiser_sinc_filter1d
from kandinsky6_mlx.utils.compare import compare
from tests.parity.audio_cases import tiny_vocoder, to_mlx_layout, vocoder_results

CONFIGS = Path(__file__).resolve().parents[2] / "fixtures" / "configs"
RESULTS = vocoder_results()
CLASSES = {
    "ConvTranspose1d",
    "MMAudioUpSample1d",
    "MMAudioSnakeBeta",
    "MMAudioDownSample1d",
    "MMAudioActivation1d",
    "MMAudioAMPBlock",
}


def test_every_vocoder_class_is_covered():
    assert {r["class"] for r in RESULTS} == CLASSES


@pytest.mark.parametrize("result", RESULTS, ids=[f"{r['module']}-{i}" for i, r in enumerate(RESULTS)])
def test_recorded_call_matches_reference(result):
    assert result["rel_max"] <= 1e-4, result


def test_forward_matches_reference():
    model, tensors, _ = tiny_vocoder()
    got = model(tensors["inputs.forward.mel"])
    assert compare(got, tensors["outputs.forward.sample"])["rel_max"] <= 1e-4


@pytest.mark.parametrize(("cutoff", "half_width", "kernel"), [(0.25, 0.3, 12), (0.5, 0.6, 12), (0.25, 0.3, 11)])
def test_initial_filter_matches_reference(cutoff, half_width, kernel):
    pytest.importorskip("torch")
    from diffusers.models.autoencoders.mmaudio_vocoder import kaiser_sinc_filter1d as reference

    assert (
        compare(kaiser_sinc_filter1d(cutoff, half_width, kernel), reference(cutoff, half_width, kernel))["rel_max"]
        <= 1e-6
    )


def test_released_config_builds_reference_parameters():
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder as Reference

    config = json.loads((CONFIGS / "lite-distill-vocoder.json").read_text())
    with torch.device("meta"):
        reference = Reference.from_config(config)
    expected = {}
    for name, value in reference.state_dict().items():
        rule = conv_rule("vocoder", name)
        shape = tuple(value.shape)
        expected[name] = tuple(shape[a] for a in rule.axes()) if rule else shape
    got = {name: tuple(v.shape) for name, v in tree_flatten(MMAudioVocoder.from_config(config).parameters())}
    assert got == expected


@pytest.mark.parametrize("mel_frames", [90, 436], ids=["smoke", "baseline"])
def test_released_rates_lengths_and_values(mel_frames):
    """Released upsampling (x512 over six stages, kernels 16/8/4/4/4/4, AMP kernels 3/7/11) at a small width."""
    torch = pytest.importorskip("torch")
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder as Reference

    from reference.make_tiny_fixtures import build

    config = {"num_mels": 16, "upsample_initial_channel": 128}
    reference = build(Reference, config)
    model = MMAudioVocoder.from_config(config)
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    model.load_weights(list(to_mlx_layout("vocoder", weights).items()), strict=True)
    mel = np.random.default_rng(0).standard_normal((1, 16, mel_frames)).astype(np.float32)
    with torch.no_grad():
        ref = reference(torch.from_numpy(mel), return_dict=False)[0].numpy()
    got = model(mx.array(mel))
    assert got.shape == ref.shape == (1, 1, 512 * mel_frames)
    assert compare(got, ref)["rel_max"] <= 1e-4
