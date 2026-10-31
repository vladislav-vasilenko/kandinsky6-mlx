"""Spec 12: image preprocessing and the HunyuanVideo VAE encoder against diffusers 0.42.1."""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from kandinsky6_mlx.models.video_vae.encoder import HunyuanVideoVAEEncoder
from kandinsky6_mlx.pipelines.image import preprocess_image
from kandinsky6_mlx.utils.compare import compare
from tests.parity.audio_cases import to_mlx_layout

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("K6_LITE_WEIGHTS", ROOT / "weights" / "lite-distill"))
I2V_INPUT = ROOT / "fixtures" / "real_inputs" / "i2v_input.png"


def synthetic_image(width: int, height: int, mode: str = "RGB", seed: int = 0) -> Image.Image:
    """A fixed image with gradients and noise, so resampling differences would show."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), (x + y) % 256], -1)
    pixels = np.clip(base + rng.integers(-30, 30, base.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    if mode == "RGBA":
        image = image.convert("RGBA")
        image.putalpha(Image.fromarray((x * 255 // max(width - 1, 1)).astype(np.uint8)))
    return image


@pytest.mark.parametrize(
    ("size", "mode", "target"),
    [
        ((640, 360), "RGB", (256, 432)),  # wider than the target: crop left/right
        ((300, 500), "RGB", (256, 432)),  # portrait: crop top/bottom
        ((432, 256), "RGB", (256, 432)),  # exact size
        ((1000, 700), "RGB", (480, 864)),  # Baseline
        ((500, 400), "RGBA", (256, 432)),  # alpha channel, as PIL handles it in the reference
        ((333, 333), "RGB", (250, 430)),  # target rounded down to multiples of 8
    ],
)
def test_preprocessing_matches_video_processor(size, mode, target):
    pytest.importorskip("diffusers")
    from diffusers.video_processor import VideoProcessor

    image = synthetic_image(*size, mode=mode)
    height, width = target
    reference = VideoProcessor(vae_scale_factor=8).preprocess(image, height=height, width=width, resize_mode="crop")
    np.testing.assert_array_equal(preprocess_image(image, height, width), reference.numpy())


def _small_structure():
    import torch
    from diffusers import AutoencoderKLHunyuanVideo as Reference

    from reference.make_tiny_fixtures import build

    config = {"block_out_channels": (8, 8, 16, 16), "norm_num_groups": 4, "layers_per_block": 2}
    reference = build(Reference, config)
    model = HunyuanVideoVAEEncoder.from_config(config)
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    model.load_weights(HunyuanVideoVAEEncoder.encoder_weights(to_mlx_layout("vae", weights)), strict=True)
    return torch, reference, model


@pytest.mark.parametrize("frames", [1, 5], ids=["image", "5-frames"])
def test_encoder_matches_reference(frames):
    """Released structure (4 down blocks, x8 spatial, x4 temporal, mid attention) at a small width; the latent
    sample uses the same NumPy noise on both sides."""
    pytest.importorskip("torch")
    torch, reference, model = _small_structure()
    rng = np.random.default_rng(frames)
    x = rng.uniform(-1, 1, (1, 3, frames, 40, 56)).astype(np.float32)
    with torch.no_grad():
        dist = reference.encode(torch.from_numpy(x)).latent_dist
        noise = rng.standard_normal(tuple(dist.mean.shape)).astype(np.float32)
        ref_sample = (dist.mean + dist.std * torch.from_numpy(noise)).numpy()
    mean, logvar = model.encode(mx.array(x))
    assert mean.shape == tuple(dist.mean.shape) == (1, 16, (frames - 1) // 4 + 1, 5, 7)
    assert compare(mean, dist.mean)["rel_max"] <= 1e-4
    assert compare(logvar, dist.logvar)["rel_max"] <= 1e-4
    sample = HunyuanVideoVAEEncoder.sample(mean, logvar, mx.array(noise))
    assert compare(sample, ref_sample)["rel_max"] <= 1e-4


def load_real_encoder(weights: Path = WEIGHTS) -> HunyuanVideoVAEEncoder:
    encoder = HunyuanVideoVAEEncoder.from_config(json.loads((weights / "vae" / "config.json").read_text()))
    stored = mx.load(str(weights / "vae" / "model.safetensors"))
    encoder.load_weights(HunyuanVideoVAEEncoder.encoder_weights(stored), strict=True)
    return encoder


@pytest.mark.real_weights
def test_real_encoder_on_a_fixed_image():
    """Real Lite VAE encoder in float32 on fixtures/real_inputs/i2v_input.png at Smoke size vs diffusers float32."""
    torch = pytest.importorskip("torch")
    if not (WEIGHTS / "vae" / "model.safetensors").is_file() or not I2V_INPUT.is_file():
        pytest.skip("needs the converted VAE and the I2V input image")
    from reference.record_video_real import load_reference

    x = preprocess_image(Image.open(I2V_INPUT), 256, 432)[:, :, None]
    reference = load_reference(WEIGHTS / "vae")
    with torch.no_grad():
        dist = reference.encode(torch.from_numpy(x)).latent_dist
    mean, logvar = load_real_encoder().encode(mx.array(x))
    assert compare(mean, dist.mean)["rel_l2"] <= 1e-3
    assert compare(logvar, dist.logvar)["rel_l2"] <= 1e-3
