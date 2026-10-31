"""Spec 13 stage B: the non-distilled Lite-5s (FlowMatch Euler, 50 steps, guidance 5, the default negative prompt)
on real weights against recorded diffusers runs (fixtures/real/base_smoke_seed{42,1}_*), Smoke size.

Fifty guided steps amplify any rounding difference about 1.3-1.5x per step: two float32 runs of torch itself (CPU and
MPS) end at latent rel_l2 1.9e-3, frames 64.7 dB and audio 38.0 dB apart (report 13). The thresholds below are derived
in report 13 from seeds 42 and 1:
- float32 (the port check): steps 0-9, before the amplification builds up, rel_l2 <= 10x the worst observed (3.8e-7);
  final frames and audio >= the worst observed minus 20 dB (63.5 and 44.6 dB).
- bf16: AGENTS.md (rel_l2 <= 2x the torch bf16 drift per step); the minimum per-step cosine not below torch bf16's
  own (0.9748 and 0.9884); frames within 1 dB of torch bf16's.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture
from tests.e2e.test_pipeline_real import stage_b_metrics

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights" / "lite-base"
SEEDS = (42, 1)
FP32_EARLY_STEPS = 10
FP32_EARLY_MAX_REL_L2 = 3.8e-6
FP32_MIN_FRAMES_PSNR_DB = 43.5
FP32_MIN_AUDIO_SNR_DB = 24.6


def files(seed: int) -> tuple[Path, Path, Path, Path]:
    stem = ROOT / "fixtures" / "real" / f"base_smoke_seed{seed}"
    return tuple(Path(f"{stem}_{part}") for part in ("fp32", "bf16", "frames", "audio"))


def run(seed: int, dtype: mx.Dtype) -> dict:
    reference, drift, frames, audio = files(seed)
    needed = [WEIGHTS / "transformer" / "model.safetensors", *(p.with_suffix(".safetensors") for p in files(seed))]
    missing = [str(f) for f in needed if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")
    result = stage_b_metrics(WEIGHTS, reference, drift, dtype=dtype, frames_fixture=frames, audio_fixture=audio)
    result.pop("_outputs")
    return result


@pytest.mark.real_weights
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_lite_base_float32(seed):
    result = run(seed, mx.float32)
    assert len(result["steps"]) == 50
    for i, entry in enumerate(result["steps"][:FP32_EARLY_STEPS]):
        for name, m in entry.items():
            assert m["rel_l2"] <= FP32_EARLY_MAX_REL_L2, (i, name, m)
    assert result["frames_psnr_db"] >= FP32_MIN_FRAMES_PSNR_DB, result["frames_psnr_db"]
    assert result["audio_snr_db"] >= FP32_MIN_AUDIO_SNR_DB, result["audio_snr_db"]


@pytest.mark.real_weights
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_lite_base_bf16(seed):
    result = run(seed, mx.bfloat16)
    reference = load_fixture(files(seed)[0], backend="numpy")[0]
    drift = load_fixture(files(seed)[1], backend="numpy")[0]
    torch_min_cosine = min(
        compare(drift[f"outputs.step.{i}.{n}"], reference[f"outputs.step.{i}.{n}"])["cosine"]
        for i in range(50)
        for n in ("latents", "audio_latents")
    )
    for i, entry in enumerate(result["steps"]):
        for name, m in entry.items():
            assert m["rel_l2"] <= 2 * m["torch_bf16_rel_l2"], (i, name, m)
    assert min(m["cosine"] for e in result["steps"] for m in e.values()) >= torch_min_cosine
    assert result["frames_psnr_db"] >= result["torch_bf16_frames_psnr_db"] - 1
