"""Spec 08 stage B: MLX video VAE decode on the real Lite-distill weights in float32.

The reference comes from `reference/record_video_real.py` (`fixtures/real/video_smoke`); the converted weights from
`weights/lite-distill/vae` (override the root with `K6_LITE_WEIGHTS`). Criterion: `rel_l2 <= 1e-3` and PSNR >= 40 dB
on frames mapped to [0, 1]. Running this file as a script prints the metrics (with peak memory and time) and writes
`outputs/08/mlx.mp4`, `outputs/08/reference.mp4` and `outputs/08/frames.png` (frames 0, 12, 24; MLX top, reference
bottom):

    uv run python -m tests.parity.test_video_vae_real
"""

import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.models.video_vae.decoder import AutoencoderKLHunyuanVideo
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("K6_LITE_WEIGHTS", ROOT / "weights" / "lite-distill"))
FIXTURE = ROOT / "fixtures" / "real" / "video_smoke"


def load_vae(weights: Path) -> AutoencoderKLHunyuanVideo:
    vae = AutoencoderKLHunyuanVideo.from_config(json.loads((weights / "vae" / "config.json").read_text()))
    stored = mx.load(str(weights / "vae" / "model.safetensors"))
    vae.load_weights(AutoencoderKLHunyuanVideo.decoder_weights(stored), strict=True)
    return vae


def decode_video(vae: AutoencoderKLHunyuanVideo, latents: mx.array) -> mx.array:
    """Transformer-layout latents `(B, T, H, W, C)` -> frames `(B, 3, F, H, W)`, as the pipeline does."""
    frames = vae.decode(latents.transpose(0, 4, 1, 2, 3) / vae.scaling_factor)
    mx.eval(frames)
    return frames


def psnr_db(actual, reference) -> float:
    """PSNR of frames mapped from [-1, 1] to [0, 1] (the pipeline's postprocessing range)."""
    a = np.clip((np.asarray(actual, np.float64) + 1) / 2, 0, 1)
    r = np.clip((np.asarray(reference, np.float64) + 1) / 2, 0, 1)
    return 10 * math.log10(1.0 / max(np.mean((a - r) ** 2), 1e-300))


def stage_b_metrics(weights: Path, fixture: Path) -> dict:
    tensors, meta = load_fixture(fixture, backend="mlx")
    vae = load_vae(weights)
    mx.reset_peak_memory()
    started = time.time()
    frames = decode_video(vae, tensors["inputs.latents"])
    seconds = time.time() - started
    reference = tensors["outputs.frames"]
    keys = ("rel_max", "rel_l2", "cosine")
    return {
        "latents": meta["latents"],
        "shapes": {"frames": list(frames.shape), "reference": list(reference.shape)},
        "frames": {k: compare(frames, reference)[k] for k in keys} | {"psnr_db": psnr_db(frames, reference)},
        "mlx_seconds": round(seconds, 2),
        "mlx_peak_memory_gib": round(mx.get_peak_memory() / 2**30, 2),
        "torch_cpu_seconds": meta.get("torch_cpu_seconds"),
        "arrays": (np.array(frames), np.array(reference)),
    }


def to_uint8(frames: np.ndarray) -> np.ndarray:
    """`(3, F, H, W)` in [-1, 1] -> `(F, H, W, 3)` uint8, like the pipeline's postprocessing."""
    return (np.clip((frames.transpose(1, 2, 3, 0) + 1) / 2, 0, 1) * 255).round().astype(np.uint8)


def write_mp4(path: Path, frames: np.ndarray, fps: int = 24) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "12"}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def write_side_by_side(path: Path, mlx_frames: np.ndarray, ref_frames: np.ndarray, indices=(0, 12, 24)) -> None:
    from PIL import Image

    indices = [i for i in indices if i < len(mlx_frames)]
    rows = [np.concatenate([frames[i] for i in indices], axis=1) for frames in (mlx_frames, ref_frames)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def _require_real():
    files = [WEIGHTS / "vae" / "model.safetensors", FIXTURE.with_suffix(".safetensors")]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")


@pytest.mark.real_weights
def test_real_video_decode_float32():
    _require_real()
    metrics = stage_b_metrics(WEIGHTS, FIXTURE)
    assert metrics["shapes"]["frames"] == metrics["shapes"]["reference"]
    assert metrics["frames"]["rel_l2"] <= 1e-3, metrics["frames"]
    assert metrics["frames"]["psnr_db"] >= 40, metrics["frames"]


def test_stage_b_harness_on_tiny_weights(tmp_path):
    """record_video_real.py on a tiny VAE saved and converted like the real one, then the MLX side (framewise path)."""
    pytest.importorskip("torch")
    from diffusers import AutoencoderKLHunyuanVideo as Reference

    from kandinsky6_mlx.convert.convert import convert_component
    from reference import record_video_real, tiny_configs
    from reference.make_tiny_fixtures import build

    build(Reference, tiny_configs.VIDEO_VAE).save_pretrained(tmp_path / "ckpt" / "vae")
    convert_component(tmp_path / "ckpt" / "vae", "vae", tmp_path / "converted" / "vae")
    out = tmp_path / "video_smoke"
    args = ["--weights", str(tmp_path / "converted"), "--shape", "1", "7", "2", "3", "16", "--out", str(out)]
    assert record_video_real.main(args) == 0
    metrics = stage_b_metrics(tmp_path / "converted", out)
    assert metrics["shapes"]["frames"] == metrics["shapes"]["reference"] == [1, 3, 25, 16, 24]
    assert metrics["frames"]["rel_max"] <= 1e-4


if __name__ == "__main__":
    result = stage_b_metrics(WEIGHTS, FIXTURE)
    mlx_frames, ref_frames = (to_uint8(a[0]) for a in result.pop("arrays"))
    out = ROOT / "outputs" / "08"
    write_mp4(out / "mlx.mp4", mlx_frames)
    write_mp4(out / "reference.mp4", ref_frames)
    write_side_by_side(out / "frames.png", mlx_frames, ref_frames)
    print(json.dumps(result, indent=2))
