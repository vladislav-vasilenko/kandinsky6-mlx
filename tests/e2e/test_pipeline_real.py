"""Spec 09 stage B: the MLX pipeline in bf16 on the real Lite-distill weights against recorded diffusers runs.

Needs `weights/lite-distill` (override with `K6_LITE_WEIGHTS`) and two recordings from
`reference/record_pipeline_run.py` on the same text embeddings and noise: `fixtures/real/pipeline_smoke_fp32` (the
reference) and `fixtures/real/pipeline_smoke_bf16` (diffusers with a bf16 transformer, the drift basis). Criteria, as
relaxed in report 09 because diffusers in bf16 itself misses the spec's absolute thresholds: per-step latent rel_l2
<= 2 x the torch bf16 drift at that step and cosine >= 0.99 (the spec's stop condition); final frames PSNR within
1 dB of torch bf16's (>= 30 dB without a drift basis); the audio SNR is reported. The port itself is checked by
`test_real_smoke_pipeline_fp32` with thresholds derived from four seeds (see FP32_*). Running this file as a script prints the metrics and writes `outputs/09/mlx.mp4` and
`outputs/09/reference.mp4`:

    uv run python -m tests.e2e.test_pipeline_real
"""

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.pipelines.export import write_mp4
from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("K6_LITE_WEIGHTS", ROOT / "weights" / "lite-distill"))
REFERENCE = ROOT / "fixtures" / "real" / "pipeline_smoke_fp32"
DRIFT = ROOT / "fixtures" / "real" / "pipeline_smoke_bf16"
# Reference decodes of REFERENCE's final latents (record_video_real.py / record_audio_real.py --latents).
FRAMES = ROOT / "fixtures" / "real" / "pipeline_smoke_frames"
AUDIO = ROOT / "fixtures" / "real" / "pipeline_smoke_audio"

# Float32 MLX pipeline vs the float32 diffusers pipeline: the check of the port itself. Derived in report 09 from
# seeds 42, 1, 2, 3 (worst observed: latent rel_l2 7.2e-5, frames 96.6 dB, audio 84.2 dB): 10x the worst rel_l2,
# and the worst PSNR/SNR minus 20 dB. rel_max is reported only: over 10 steps single elements reach 9e-4, above the
# AGENTS.md per-block 1e-4, while rel_l2 stays below 1e-4.
FP32_MAX_REL_L2 = 7.2e-4
FP32_MIN_FRAMES_PSNR_DB = 76.0
FP32_MIN_AUDIO_SNR_DB = 64.0
SEEDS = (42, 1, 2, 3)


def seed_fixtures(seed: int) -> tuple[Path, Path, Path, Path]:
    """Reference, bf16 drift, reference frames and audio recorded for `seed` (seed 42 has the original names)."""
    stem = "pipeline_smoke" if seed == 42 else f"pipeline_smoke_seed{seed}"
    root = ROOT / "fixtures" / "real"
    return root / f"{stem}_fp32", root / f"{stem}_bf16", root / f"{stem}_frames", root / f"{stem}_audio"


def reference_media(ref: dict, frames: Path | None, audio: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Reference frames `(B, F, H, W, 3)` in [0, 1] and waveform `(B, samples)`: from the pipeline recording itself
    (`output_type="np"`) or from the separate decoder recordings, postprocessed like the pipeline."""
    if frames is None:
        return ref["outputs.frames"], ref["outputs.audio"]
    video = load_fixture(frames, backend="numpy")[0]["outputs.frames"]  # (B, 3, F, H, W) in [-1, 1]
    video = np.clip(video / 2 + 0.5, 0, 1).transpose(0, 2, 3, 4, 1)
    waveform = load_fixture(audio, backend="numpy")[0]["outputs.waveform"][:, 0]
    return video, waveform


def psnr_db(a, b) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-300))


def snr_db(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 10 * math.log10(np.sum(b**2) / max(np.sum((a - b) ** 2), 1e-300))


def stage_b_metrics(
    weights: Path,
    reference: Path,
    drift: Path | None,
    dtype: mx.Dtype = mx.bfloat16,
    frames_fixture: Path | None = None,
    audio_fixture: Path | None = None,
) -> dict:
    ref, meta = load_fixture(reference, backend="numpy")
    ref_frames, ref_audio = reference_media(ref, frames_fixture, audio_fixture)
    basis = load_fixture(drift, backend="numpy")[0] if drift is not None else None
    pipeline = Kandinsky6Pipeline.from_converted(weights, dtype)
    output = pipeline(
        {
            "encoder_hidden_states": mx.array(ref["inputs.encoder_hidden_states"]),
            "pooled_projections": mx.array(ref["inputs.pooled_projections"]),
        },
        num_inference_steps=meta["steps"],
        latents=ref["inputs.latents"],
        audio_latents=ref["inputs.audio_latents"],
        return_step_latents=True,
        release_transformer=True,
    )
    steps = []
    for i, (video, audio) in enumerate(output.step_latents):
        # diffusers keeps video latents as (B, T, H, W, C) during the loop, like the MLX pipeline.
        entry = {}
        for name, got in (("video", video), ("audio", audio)):
            key = f"outputs.step.{i}.{'latents' if name == 'video' else 'audio_latents'}"
            metrics = compare(got, ref[key])
            entry[name] = {"rel_l2": metrics["rel_l2"], "cosine": metrics["cosine"], "rel_max": metrics["rel_max"]}
            if basis is not None:
                entry[name]["torch_bf16_rel_l2"] = compare(basis[key], ref[key])["rel_l2"]
        steps.append(entry)
    result = {
        "steps": steps,
        "frames_psnr_db": psnr_db(output.frames, ref_frames),
        "audio_snr_db": snr_db(output.audio, ref_audio),
        "seconds": {k: round(v, 2) for k, v in output.timings.items()},
        "peak_memory_gib": {k: round(v, 2) for k, v in output.peak_memory_gib.items()},
        "reference_seconds": meta.get("seconds"),
        "_outputs": (output, ref_frames, ref_audio),
    }
    if basis is not None:
        if "outputs.frames" in basis:
            torch_frames = basis["outputs.frames"]
        else:  # latent-only recording: decode its last step with the (float32, verified) MLX VAE
            last = mx.array(basis[f"outputs.step.{len(steps) - 1}.latents"]).transpose(0, 4, 1, 2, 3)
            decoded = pipeline.vae.decode(last / pipeline.vae.scaling_factor)
            torch_frames = np.array(mx.clip(decoded / 2 + 0.5, 0, 1).transpose(0, 2, 3, 4, 1))
        result["torch_bf16_frames_psnr_db"] = psnr_db(torch_frames, ref_frames)
    return result


def violations(result: dict) -> list[dict]:
    failed = []
    for i, entry in enumerate(result["steps"]):
        for name, m in entry.items():
            limit = 2 * m.get("torch_bf16_rel_l2", math.inf)
            min_cosine = 0.99 if "torch_bf16_rel_l2" in m else 0.999
            if m["cosine"] < min_cosine or m["rel_l2"] > limit:
                failed.append({"step": i, "latents": name, "limit": limit, **m})
    min_psnr = result["torch_bf16_frames_psnr_db"] - 1 if "torch_bf16_frames_psnr_db" in result else 30
    if result["frames_psnr_db"] < min_psnr:
        failed.append({"frames_psnr_db": result["frames_psnr_db"], "limit": min_psnr})
    return failed


@pytest.mark.real_weights
@pytest.mark.slow
def test_real_smoke_pipeline_bf16():
    files = [WEIGHTS / "transformer" / "model.safetensors"]
    files += [f.with_suffix(".safetensors") for f in (REFERENCE, DRIFT, FRAMES, AUDIO)]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")
    failed = violations(stage_b_metrics(WEIGHTS, REFERENCE, DRIFT, frames_fixture=FRAMES, audio_fixture=AUDIO))
    assert not failed, f"first failure: {failed[0]}"


@pytest.mark.real_weights
@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_real_smoke_pipeline_fp32(seed):
    """The port check: float32 MLX vs float32 diffusers from the same noise, tight thresholds derived in report 09."""
    reference, _, frames, audio = seed_fixtures(seed)
    files = [WEIGHTS / "transformer" / "model.safetensors"]
    files += [f.with_suffix(".safetensors") for f in (reference, frames, audio)]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")
    result = stage_b_metrics(WEIGHTS, reference, None, dtype=mx.float32, frames_fixture=frames, audio_fixture=audio)
    worst = max((m["rel_l2"], i, name) for i, entry in enumerate(result["steps"]) for name, m in entry.items())
    assert worst[0] <= FP32_MAX_REL_L2, f"step {worst[1]} {worst[2]}: rel_l2 {worst[0]:.2e}"
    assert result["frames_psnr_db"] >= FP32_MIN_FRAMES_PSNR_DB, result["frames_psnr_db"]
    assert result["audio_snr_db"] >= FP32_MIN_AUDIO_SNR_DB, result["audio_snr_db"]


def test_stage_b_harness_on_tiny_pipeline(tiny_pipelines, tmp_path):
    """record_pipeline_run.py on the tiny pipeline's diffusers snapshot, then the MLX side in float32."""
    from kandinsky6_mlx.utils.fixtures import save_fixture
    from reference import record_pipeline_run

    torch, _, _, root = tiny_pipelines
    generator = torch.Generator().manual_seed(0)
    text = {
        "inputs.encoder_hidden_states": torch.randn(1, 6, 8, generator=generator),
        "inputs.pooled_projections": torch.randn(1, 4, generator=generator),
        "inputs.encoder_attention_mask": torch.tensor([[True, True, True, True, True, False]]),
    }
    save_fixture(tmp_path / "text", text, {"fixture": "text"})
    out = tmp_path / "pipeline"
    args = [
        "--checkpoint",
        str(root / "ckpt"),
        "--text-embeds",
        str(tmp_path / "text"),
        "--steps",
        "3",
        "--out",
        str(out),
    ]
    assert record_pipeline_run.main(args) == 0
    result = stage_b_metrics(root / "converted", out, out, dtype=mx.float32)
    assert len(result["steps"]) == 3
    assert result["frames_psnr_db"] > 60 and result["audio_snr_db"] > 60
    for entry in result["steps"]:
        for m in entry.values():
            assert m["cosine"] > 0.99999 and m["rel_l2"] < 1e-4


if __name__ == "__main__":
    drift = DRIFT if DRIFT.with_suffix(".safetensors").is_file() else None
    result = stage_b_metrics(WEIGHTS, REFERENCE, drift, frames_fixture=FRAMES, audio_fixture=AUDIO)
    output, ref_frames, ref_audio = result.pop("_outputs")
    out = ROOT / "outputs" / "09"
    write_mp4(out / "mlx.mp4", output.frames[0], output.frame_rate, output.audio[0], output.sample_rate)
    write_mp4(out / "reference.mp4", ref_frames[0], output.frame_rate, ref_audio[0], output.sample_rate)
    result["violations"] = violations(result)
    print(json.dumps(result, indent=2))
