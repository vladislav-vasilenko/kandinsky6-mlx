"""Spec 09 stage B: the MLX pipeline in bf16 on the real Lite-distill weights against recorded diffusers runs.

Needs `weights/lite-distill` (override with `K6_LITE_WEIGHTS`) and two recordings from
`reference/record_pipeline_run.py` on the same text embeddings and noise: `fixtures/real/pipeline_smoke_fp32` (the
reference) and `fixtures/real/pipeline_smoke_bf16` (diffusers with a bf16 transformer, the drift basis). Criteria:
per-step latent cosine >= 0.999 and rel_l2 <= 2 x the torch bf16 drift at that step; final frames PSNR >= 30 dB; the
audio SNR is reported. Running this file as a script prints the metrics and writes `outputs/09/mlx.mp4` and
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


def psnr_db(a, b) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-300))


def snr_db(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 10 * math.log10(np.sum(b**2) / max(np.sum((a - b) ** 2), 1e-300))


def stage_b_metrics(weights: Path, reference: Path, drift: Path | None, dtype: mx.Dtype = mx.bfloat16) -> dict:
    ref, meta = load_fixture(reference, backend="numpy")
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
            entry[name] = {"rel_l2": metrics["rel_l2"], "cosine": metrics["cosine"]}
            if basis is not None:
                entry[name]["torch_bf16_rel_l2"] = compare(basis[key], ref[key])["rel_l2"]
        steps.append(entry)
    result = {
        "steps": steps,
        "frames_psnr_db": psnr_db(output.frames, ref["outputs.frames"]),
        "audio_snr_db": snr_db(output.audio, ref["outputs.audio"]),
        "seconds": {k: round(v, 2) for k, v in output.timings.items()},
        "peak_memory_gib": {k: round(v, 2) for k, v in output.peak_memory_gib.items()},
        "reference_seconds": meta.get("seconds"),
        "_outputs": (output, ref),
    }
    if basis is not None:
        result["torch_bf16_frames_psnr_db"] = psnr_db(basis["outputs.frames"], ref["outputs.frames"])
    return result


def violations(result: dict) -> list[dict]:
    failed = []
    for i, entry in enumerate(result["steps"]):
        for name, m in entry.items():
            limit = 2 * m.get("torch_bf16_rel_l2", math.inf)
            if m["cosine"] < 0.999 or m["rel_l2"] > limit:
                failed.append({"step": i, "latents": name, "limit": limit, **m})
    if result["frames_psnr_db"] < 30:
        failed.append({"frames_psnr_db": result["frames_psnr_db"]})
    return failed


@pytest.mark.real_weights
@pytest.mark.slow
def test_real_smoke_pipeline_bf16():
    files = [WEIGHTS / "transformer" / "model.safetensors", REFERENCE.with_suffix(".safetensors")]
    files.append(DRIFT.with_suffix(".safetensors"))
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")
    failed = violations(stage_b_metrics(WEIGHTS, REFERENCE, DRIFT))
    assert not failed, f"first failure: {failed[0]}"


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
    result = stage_b_metrics(WEIGHTS, REFERENCE, drift)
    output, ref = result.pop("_outputs")
    out = ROOT / "outputs" / "09"
    write_mp4(out / "mlx.mp4", output.frames[0], output.frame_rate, output.audio[0], output.sample_rate)
    write_mp4(
        out / "reference.mp4", ref["outputs.frames"][0], output.frame_rate, ref["outputs.audio"][0], output.sample_rate
    )
    result["violations"] = violations(result)
    print(json.dumps(result, indent=2))
