"""Spec 10: tiled video VAE decoding against diffusers `enable_tiling` (`tiled_decode`, `_temporal_tiled_decode`)."""

from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.models.video_vae.decoder import TilingConfig, blend
from kandinsky6_mlx.utils.compare import compare
from tests.parity.test_video_vae_decoder import _small_released_structure

# Small tiles for small latents: 4x4 latent tiles with stride 3 (32/24 px, blend 8 px); temporal tiles of 2 latent
# frames + 1 with stride 1 (8/4 frames, blend 4). Latent sizes 7 and 10 are not multiples of the stride.
ROOT = Path(__file__).resolve().parents[2]
SMALL = TilingConfig(32, 32, 8, 24, 24, 4)
SPATIAL_ONLY = TilingConfig(32, 32, 16, 24, 24, 12)
TEMPORAL_ONLY = TilingConfig(1024, 1024, 8, 768, 768, 4)

CASES = {
    "spatial-only": (SPATIAL_ONLY, (2, 7, 10)),
    "temporal-only": (TEMPORAL_ONLY, (7, 4, 6)),
    "spatial+temporal": (SMALL, (7, 7, 10)),
    "default-tiles-one-spatial-tile": (TilingConfig(), (7, 4, 6)),
    "wide-only": (SMALL, (3, 3, 11)),
}


@pytest.mark.parametrize("case", CASES)
def test_tiled_decode_matches_reference(case):
    pytest.importorskip("torch")
    torch, reference, model = _small_released_structure()
    tiling, (frames, height, width) = CASES[case]
    reference.enable_tiling(**asdict(tiling))
    z = np.random.default_rng(len(case)).standard_normal((1, 16, frames, height, width)).astype(np.float32)
    with torch.no_grad():
        ref = reference.decode(torch.from_numpy(z), return_dict=False)[0].numpy()
    got = model.decode(mx.array(z), tiling=tiling)
    assert got.shape == ref.shape == (1, 3, 4 * (frames - 1) + 1, 8 * height, 8 * width)
    assert compare(got, ref)["rel_max"] <= 1e-4


def test_tiling_changes_the_output():
    """Guards the test above: per-tile GroupNorm statistics make tiled and untiled decodes differ."""
    pytest.importorskip("torch")
    _, _, model = _small_released_structure()
    z = mx.array(np.random.default_rng(3).standard_normal((1, 16, 2, 7, 10)).astype(np.float32))
    assert compare(model.decode(z, tiling=SPATIAL_ONLY), model.decode(z))["rel_max"] > 1e-3


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_blend_matches_reference_loops(axis):
    torch = pytest.importorskip("torch")
    from diffusers import AutoencoderKLHunyuanVideo as Reference

    rng = np.random.default_rng(axis)
    a, b = (rng.standard_normal((1, 6, 5, 7, 3)).astype(np.float32) for _ in range(2))
    reference_blend = {1: Reference.blend_t, 2: Reference.blend_v, 3: Reference.blend_h}[axis]
    # The reference works on (B, C, T, H, W); the MLX decoder blends channels-last (B, T, H, W, C).
    to_ref = lambda x: torch.from_numpy(x.transpose(0, 4, 1, 2, 3).copy())
    expected = reference_blend(None, to_ref(a), to_ref(b), 4).numpy().transpose(0, 2, 3, 4, 1)
    np.testing.assert_array_equal(np.array(blend(mx.array(a), mx.array(b), 4, axis)), expected)


@pytest.mark.parametrize("case", [*CASES, "no-tiling-T7", "no-tiling-T2"])
def test_tile_callback_count_matches_num_tiles(case):
    """Progress reporting: `decode` calls back once per tile and `num_tiles` predicts how many."""
    pytest.importorskip("torch")
    _, _, model = _small_released_structure()
    if case in CASES:
        tiling, (frames, height, width) = CASES[case]
    else:
        tiling, (frames, height, width) = None, (int(case[-1]), 4, 6)
    shape = (1, 16, frames, height, width)
    calls = []
    model.decode(mx.zeros(shape), tiling=tiling, callback=lambda: calls.append(1))
    assert len(calls) == model.num_tiles(shape, tiling) >= 1


# Stage B: the real Lite-distill VAE with the reference's default tiling on Baseline latents (480x864) from a real
# MLX generation (scripts/generate.py --preset baseline --save-latents). The torch CPU reference peaks near 40 GB and
# needs about 4 minutes per temporal tile, so it decodes the first 8 latent frames (29 frames): three temporal tiles
# (the last one partial), each a 3 x 5 grid of spatial tiles with partial edge tiles.
REAL_WEIGHTS = ROOT / "weights" / "lite-distill"
REAL_REFERENCE = ROOT / "fixtures" / "real" / "baseline_tiled_ref8"


def stage_b_metrics(weights=REAL_WEIGHTS, fixture=REAL_REFERENCE) -> dict:
    import time

    from kandinsky6_mlx.utils.fixtures import load_fixture
    from tests.parity.test_video_vae_real import load_vae, psnr_db

    tensors, meta = load_fixture(fixture, backend="mlx")
    vae = load_vae(weights)
    z = tensors["inputs.latents"].transpose(0, 4, 1, 2, 3) / vae.scaling_factor
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    frames = vae.decode(z, tiling=TilingConfig())
    mx.eval(frames)
    seconds = time.perf_counter() - start
    ref = tensors["outputs.frames"]
    a, r = np.array(frames), np.array(ref)
    return {
        "shape": list(frames.shape),
        "frames": compare(frames, ref),
        "psnr_db": psnr_db(a, r),
        "per_frame_psnr_db": [round(psnr_db(a[:, :, f], r[:, :, f]), 1) for f in range(a.shape[2])],
        "mlx_seconds": round(seconds, 1),
        "mlx_peak_memory_gib": round(mx.get_peak_memory() / 2**30, 2),
        "torch_cpu_seconds": meta.get("torch_cpu_seconds"),
    }


@pytest.mark.real_weights
def test_real_baseline_tiled_decode():
    files = [REAL_WEIGHTS / "vae" / "model.safetensors", REAL_REFERENCE.with_suffix(".safetensors")]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")
    result = stage_b_metrics()
    assert result["psnr_db"] >= 40, result["psnr_db"]


if __name__ == "__main__":
    import json

    print(json.dumps(stage_b_metrics(), indent=2))
