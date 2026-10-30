"""Spec 07 stage B: MLX audio decode (latents -> mel -> waveform) on the real Lite-distill weights in float32.

The reference comes from `reference/record_audio_real.py` (`fixtures/real/audio_smoke`); the converted weights from
`weights/lite-distill/{audio_vae,vocoder}` (override the root with `K6_LITE_WEIGHTS`). Criterion: waveform
`rel_l2 <= 1e-3`; the SNR is reported. Running this file as a script prints the metrics and writes
`outputs/07/mlx.wav` and `outputs/07/reference.wav`:

    uv run python -m tests.parity.test_audio_real
"""

import json
import math
import os
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.models.audio_vae.decoder import MMAudioVAE
from kandinsky6_mlx.models.vocoder.bigvgan import MMAudioVocoder
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = Path(os.environ.get("K6_LITE_WEIGHTS", ROOT / "weights" / "lite-distill"))
FIXTURE = ROOT / "fixtures" / "real" / "audio_smoke"


def load_audio(weights: Path) -> tuple[MMAudioVAE, MMAudioVocoder]:
    audio_vae = MMAudioVAE.from_config(json.loads((weights / "audio_vae" / "config.json").read_text()))
    stored = mx.load(str(weights / "audio_vae" / "model.safetensors"))
    audio_vae.load_weights(MMAudioVAE.decoder_weights(stored), strict=True)
    vocoder = MMAudioVocoder.from_config(json.loads((weights / "vocoder" / "config.json").read_text()))
    vocoder.load_weights(str(weights / "vocoder" / "model.safetensors"), strict=True)
    return audio_vae, vocoder


def decode_audio(audio_vae: MMAudioVAE, vocoder: MMAudioVocoder, latents: mx.array) -> tuple[mx.array, mx.array]:
    """DiT-layout latents `(B, L, C)` -> mel and waveform, as the pipeline does."""
    mel = audio_vae.decode(latents.transpose(0, 2, 1) / audio_vae.scaling_factor)
    waveform = vocoder(mel)
    mx.eval(mel, waveform)
    return mel, waveform


def snr_db(actual, reference) -> float:
    a, r = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    return 10 * math.log10(np.sum(r**2) / max(np.sum((a - r) ** 2), 1e-300))


def stage_b_metrics(weights: Path, fixture: Path) -> dict:
    tensors, meta = load_fixture(fixture, backend="mlx")
    mel, waveform = decode_audio(*load_audio(weights), tensors["inputs.audio_latents"])
    ref_mel, ref_wave = tensors["outputs.mel"], tensors["outputs.waveform"]
    keys = ("rel_max", "rel_l2", "cosine")
    return {
        "latents": meta["latents"],
        "shapes": {"mel": list(mel.shape), "waveform": list(waveform.shape), "reference": list(ref_wave.shape)},
        "mel": {k: compare(mel, ref_mel)[k] for k in keys},
        "waveform": {k: compare(waveform, ref_wave)[k] for k in keys} | {"snr_db": snr_db(waveform, ref_wave)},
        "waveforms": (np.array(waveform)[0, 0], np.array(ref_wave)[0, 0], meta["sample_rate"]),
    }


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def _require_real():
    files = [WEIGHTS / c / "model.safetensors" for c in ("audio_vae", "vocoder")] + [
        FIXTURE.with_suffix(".safetensors")
    ]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        pytest.skip(f"needs {missing}")


@pytest.mark.real_weights
def test_real_audio_decode_float32():
    _require_real()
    metrics = stage_b_metrics(WEIGHTS, FIXTURE)
    assert metrics["shapes"]["waveform"] == metrics["shapes"]["reference"]
    assert metrics["waveform"]["rel_l2"] <= 1e-3, metrics["waveform"]


def test_stage_b_harness_on_tiny_weights(tmp_path):
    """record_audio_real.py on tiny models saved and converted like the real ones, then the MLX side."""
    pytest.importorskip("torch")
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE as ReferenceVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder as ReferenceVocoder

    from kandinsky6_mlx.convert.convert import convert_component
    from reference import record_audio_real, tiny_configs
    from reference.make_tiny_fixtures import build

    vae_config = tiny_configs.AUDIO_VAE | {"latent_channels": 40}
    for component, cls, config in (
        ("audio_vae", ReferenceVAE, vae_config),
        ("vocoder", ReferenceVocoder, tiny_configs.VOCODER),
    ):
        build(cls, config).save_pretrained(tmp_path / "ckpt" / component)
        convert_component(tmp_path / "ckpt" / component, component, tmp_path / "converted" / component)
    out = tmp_path / "audio_smoke"
    assert record_audio_real.main(["--weights", str(tmp_path / "converted"), "--out", str(out)]) == 0
    metrics = stage_b_metrics(tmp_path / "converted", out)
    assert metrics["shapes"]["waveform"] == metrics["shapes"]["reference"] == [1, 1, 45 * 2 * 16]
    assert metrics["mel"]["rel_max"] <= 1e-4 and metrics["waveform"]["rel_l2"] <= 1e-4


if __name__ == "__main__":
    result = stage_b_metrics(WEIGHTS, FIXTURE)
    mlx_wave, ref_wave, rate = result.pop("waveforms")
    write_wav(ROOT / "outputs" / "07" / "mlx.wav", mlx_wave, rate)
    write_wav(ROOT / "outputs" / "07" / "reference.wav", ref_wave, rate)
    result["wav_seconds"] = {"mlx": mlx_wave.size / rate, "reference": ref_wave.size / rate}
    print(json.dumps(result, indent=2))
