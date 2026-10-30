"""The recorder captures submodule calls of a reference model and replays them exactly."""

import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture, save_fixture
from reference import record_real_activations, tiny_configs
from reference.make_tiny_fixtures import build, dit_inputs
from reference.recorder import ActivationRecorder, iter_tensors, recorded_output, replay

TINY = Path(__file__).resolve().parents[2] / "fixtures" / "tiny"


@pytest.fixture(scope="module")
def tiny_dit():
    return build(diffusers.Kandinsky6Transformer3DModel, tiny_configs.DIT)


def assert_same(new, old, tol=1e-6):
    new_tensors, old_tensors = list(iter_tensors(new)), list(iter_tensors(old))
    assert len(new_tensors) == len(old_tensors) > 0
    for a, b in zip(new_tensors, old_tensors):
        assert compare(a, b)["rel_max"] <= tol


def test_replay_in_memory(tiny_dit):
    inputs = dit_inputs(np.random.default_rng(0), "av")
    names = ["visual_transformer_blocks.1", "video_text_transformer_blocks.0", "visual_rope_embeddings"]
    with ActivationRecorder(tiny_dit, names) as recorder, torch.no_grad():
        tiny_dit(**inputs)
    assert [call["module"] for call in recorder.calls] == [
        "video_text_transformer_blocks.0",
        "visual_rope_embeddings",
        "visual_transformer_blocks.1",
    ]
    for call in recorder.calls:
        assert_same(replay(tiny_dit, call, recorder.tensors), recorded_output(call, recorder.tensors))


def test_replay_from_saved_fixture(tiny_dit):
    tensors, meta = load_fixture(TINY / "dit")
    weights = {k[len("weights.") :]: torch.from_numpy(v.copy()) for k, v in tensors.items() if k.startswith("weights.")}
    model = build(diffusers.Kandinsky6Transformer3DModel, tiny_configs.DIT)
    model.load_state_dict(weights)
    calls = meta["calls"]["av"]
    blocks = [c for c in calls if re.fullmatch(r"visual_transformer_blocks\.\d+", c["module"])]
    assert len(blocks) == tiny_configs.DIT["num_visual_blocks"]
    # Every recorded primitive and block reproduces its recorded output from its recorded inputs.
    for call in calls:
        assert_same(replay(model, call, tensors), recorded_output(call, tensors))


def test_cli_on_saved_tiny_model(tmp_path, tiny_dit):
    checkpoint = tmp_path / "checkpoint"
    tiny_dit.save_pretrained(checkpoint / "transformer")
    inputs = dit_inputs(np.random.default_rng(2), "rope_pos")
    tensors = {}
    for key, value in inputs.items():
        if isinstance(value, tuple):
            tensors.update({f"inputs.{key}.{i}": v for i, v in enumerate(value)})
        else:
            tensors[f"inputs.{key}"] = value
    save_fixture(tmp_path / "inputs", tensors, {})

    out = tmp_path / "recorded"
    args = [
        "--checkpoint",
        str(checkpoint),
        "--component",
        "transformer",
        "--inputs",
        str(tmp_path / "inputs.safetensors"),
        "--modules",
        "blocks",
        "--out",
        str(out),
    ]
    assert record_real_activations.main(args) == 0

    recorded, meta = load_fixture(out)
    with torch.no_grad():
        expected = tiny_dit(**inputs, return_dict=False)
    assert compare(recorded["outputs.run.out.0"], expected[0])["rel_max"] <= 1e-6
    assert compare(recorded["outputs.run.out.1"], expected[1])["rel_max"] <= 1e-6
    calls = meta["calls"]["run"]
    assert len(calls) == tiny_configs.DIT["num_visual_blocks"] + 2 * tiny_configs.DIT["num_text_blocks"]
    for call in calls:
        assert_same(replay(tiny_dit, call, recorded), recorded_output(call, recorded))


@pytest.mark.parametrize(
    ("name", "class_name", "config", "tag"),
    [
        ("video_vae_decoder", "AutoencoderKLHunyuanVideo", tiny_configs.VIDEO_VAE, "decode"),
        ("audio_vae_decoder", "MMAudioVAE", tiny_configs.AUDIO_VAE, "decode"),
        ("vocoder", "MMAudioVocoder", tiny_configs.VOCODER, "forward"),
        ("dit_variants", None, None, None),
    ],
)
def test_every_recorded_call_replays(name, class_name, config, tag):
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    classes = {"MMAudioVAE": MMAudioVAE, "MMAudioVocoder": MMAudioVocoder}
    tensors, meta = load_fixture(TINY / name)
    if name == "dit_variants":
        runs = [
            (diffusers.Kandinsky6Transformer3DModel, dict(tiny_configs.DIT, **flags), f"weights.{variant}.", variant)
            for variant, flags in meta["variants"].items()
        ]
    else:
        runs = [(classes.get(class_name) or getattr(diffusers, class_name), config, "weights.", tag)]
    for cls, cfg, prefix, run_tag in runs:
        model = cls(**cfg).float().eval()
        state = {k[len(prefix) :]: torch.from_numpy(v.copy()) for k, v in tensors.items() if k.startswith(prefix)}
        model.load_state_dict(state)
        calls = meta["calls"][run_tag]
        assert calls
        for call in calls:
            assert_same(replay(model, call, tensors), recorded_output(call, tensors))
