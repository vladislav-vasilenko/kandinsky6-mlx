"""Spec 12 stage B: image-to-video with the real Lite-distill weights against recorded diffusers runs.

Cases (fixtures/real/i2v_<case>_{fp32,bf16,frames,audio}, from reference/record_pipeline_run.py --image): the
spec 10 Baseline first frame (fixtures/real_inputs/i2v_input.png) with the Smoke prompt at seeds 42, 1, 2, and a
tram frame (i2v_input_tram.png) with its preset prompt at seed 42. Float32 uses the port thresholds derived in report
09; bf16 uses the original spec 09 criteria (per-step cosine >= 0.999, rel_l2 <= 2 x torch bf16, frames >= 30 dB),
which image-to-video meets.
"""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from tests.e2e.test_pipeline_real import (
    FP32_MAX_REL_L2,
    FP32_MIN_AUDIO_SNR_DB,
    FP32_MIN_FRAMES_PSNR_DB,
    WEIGHTS,
    stage_b_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ("woman_seed42", "woman_seed1", "woman_seed2", "tram_seed42")


def case_files(case: str) -> tuple[Path, Path, Path, Path]:
    stem = ROOT / "fixtures" / "real" / f"i2v_{case}"
    return tuple(Path(f"{stem}_{part}") for part in ("fp32", "bf16", "frames", "audio"))


def psnr_db(a, b) -> float:
    return 10 * math.log10(
        1 / max(float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)), 1e-300)
    )


def metrics(case: str, dtype: mx.Dtype) -> dict:
    from kandinsky6_mlx.pipelines.image import preprocess_image
    from kandinsky6_mlx.utils.fixtures import load_fixture

    reference, drift, frames, audio = case_files(case)
    missing = [
        str(f)
        for f in (
            WEIGHTS / "transformer" / "model.safetensors",
            *(p.with_suffix(".safetensors") for p in case_files(case)),
        )
        if not f.is_file()
    ]
    if missing:
        pytest.skip(f"needs {missing}")
    result = stage_b_metrics(WEIGHTS, reference, drift, dtype=dtype, frames_fixture=frames, audio_fixture=audio)
    output, ref_frames, _ = result.pop("_outputs")
    image = load_fixture(reference, backend="numpy")[1]["image"]
    target = (
        preprocess_image(Image.open(ROOT / image if not Path(image).is_absolute() else image), 256, 432)[0].transpose(
            1, 2, 0
        )
        + 1
    ) / 2
    result["first_frame_psnr_vs_input_db"] = psnr_db(output.frames[0, 0], target)
    result["torch_first_frame_psnr_vs_input_db"] = psnr_db(ref_frames[0, 0], target)
    return result


@pytest.mark.real_weights
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_real_i2v_float32(case):
    result = metrics(case, mx.float32)
    worst = max((m["rel_l2"], i, name) for i, entry in enumerate(result["steps"]) for name, m in entry.items())
    assert worst[0] <= FP32_MAX_REL_L2, worst
    assert result["frames_psnr_db"] >= FP32_MIN_FRAMES_PSNR_DB and result["audio_snr_db"] >= FP32_MIN_AUDIO_SNR_DB
    assert abs(result["first_frame_psnr_vs_input_db"] - result["torch_first_frame_psnr_vs_input_db"]) <= 0.05


@pytest.mark.real_weights
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_real_i2v_bf16(case):
    result = metrics(case, mx.bfloat16)
    for i, entry in enumerate(result["steps"]):
        for name, m in entry.items():
            assert m["cosine"] >= 0.999 and m["rel_l2"] <= 2 * m["torch_bf16_rel_l2"], (i, name, m)
    assert result["frames_psnr_db"] >= 30
    # Fidelity of the first frame to the input image: not worse than the float32 reference by more than 0.5 dB.
    assert result["first_frame_psnr_vs_input_db"] >= result["torch_first_frame_psnr_vs_input_db"] - 0.5
