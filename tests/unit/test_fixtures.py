import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.utils.fixtures import load_fixture, save_fixture, select

TINY = Path(__file__).resolve().parents[2] / "fixtures" / "tiny"
EXPECTED = ["dit", "dit_variants", "video_vae_decoder", "audio_vae_decoder", "vocoder", "piflow"]


def test_round_trip(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    tensors = {
        "inputs.x": rng.standard_normal((2, 3)).astype(np.float32),
        "inputs.mask": np.array([True, False]),
        "outputs.y": rng.integers(0, 10, (4,)).astype(np.int64),
        "weights.w": rng.standard_normal((5,)).astype(np.float32),
    }
    meta = {"config": {"a": 1}, "seeds": {"numpy": 0}}
    save_fixture(tmp_path / "f", tensors, meta)
    loaded, loaded_meta = load_fixture(tmp_path / "f")
    assert loaded_meta == meta
    assert set(loaded) == set(tensors)
    for key, value in tensors.items():
        np.testing.assert_array_equal(loaded[key], value)
        assert loaded[key].dtype == value.dtype
    assert select(loaded, "inputs").keys() == {"x", "mask"}


def test_bfloat16_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    value = torch.linspace(-3, 3, 7, dtype=torch.bfloat16)
    save_fixture(tmp_path / "b", {"outputs.v": value}, {})
    as_mlx, _ = load_fixture(tmp_path / "b", backend="mlx")
    assert as_mlx["outputs.v"].dtype == mx.bfloat16
    as_numpy, _ = load_fixture(tmp_path / "b")
    np.testing.assert_array_equal(as_numpy["outputs.v"], value.float().numpy())


def test_rejects_unknown_prefix(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must start with"):
        save_fixture(tmp_path / "bad", {"other.x": np.zeros(1)}, {})


@pytest.mark.parametrize("name", EXPECTED)
def test_tiny_fixture_present_and_finite(name):
    tensors, meta = load_fixture(TINY / name)
    assert meta["fixture"] == name
    assert meta["versions"]["diffusers"] == "0.42.1"
    outputs = select(tensors, "outputs")
    assert outputs
    for key, value in outputs.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), key


def test_tiny_fixtures_size_limit():
    total = sum(path.stat().st_size for path in TINY.iterdir())
    assert total < 20 * 2**20


def test_dit_fixture_has_every_block_output():
    _, meta = load_fixture(TINY / "dit")
    for scenario in ("av", "video_only", "rope_pos"):
        modules = {call["module"] for call in meta["calls"][scenario]}
        assert {"visual_transformer_blocks.0", "visual_transformer_blocks.1"} <= modules
        assert {"video_text_transformer_blocks.0", "audio_text_transformer_blocks.0"} <= modules
    classes = {call["class"] for call in meta["calls"]["av"]}
    for required in (
        "Kandinsky6RoPE3D",
        "Kandinsky6Attention",
        "FeedForward",
        "Kandinsky6Modulation",
        "Kandinsky6VisualEmbeddings",
        "Kandinsky6OutLayer",
        "Kandinsky6OutLayerAudio",
    ):
        assert required in classes


def test_sidecar_is_valid_json():
    for name in EXPECTED:
        json.loads((TINY / f"{name}.json").read_text())
