"""The Kaggle conversion kernel is generated from the converter and works on a local snapshot layout."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "kaggle" / "convert" / "convert_k6.py"


def test_kernel_is_in_sync_with_generator():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_kaggle_convert.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_kernel_metadata():
    metadata = json.loads((ROOT / "kaggle" / "convert" / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["kernel_type"] == "script"
    assert metadata["kernel_sources"] == ["applehawk/k6-lite-diffusers-weights-media"]


def test_kernel_upload_off_and_no_token_in_source():
    source = KERNEL.read_text()
    assert "\nUPLOAD = False\n" in source
    assert "hf_" not in source.lower().replace("hf_token", "").replace("hf_repo", "")


def test_kernel_converts_a_snapshot(tmp_path):
    pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    from reference import tiny_configs
    from reference.make_tiny_fixtures import build

    snapshot = tmp_path / "input" / "notebooks" / "applehawk" / "k6-lite-diffusers-weights-media" / "k6-lite"
    for component, cls, config in [
        ("transformer", diffusers.Kandinsky6Transformer3DModel, tiny_configs.DIT),
        ("vae", diffusers.AutoencoderKLHunyuanVideo, tiny_configs.VIDEO_VAE),
        ("audio_vae", MMAudioVAE, tiny_configs.AUDIO_VAE),
        ("vocoder", MMAudioVocoder, tiny_configs.VOCODER),
    ]:
        build(cls, config).save_pretrained(snapshot / component)
    (snapshot / "weights-manifest.json").write_text(json.dumps({"revision": "4c487e8b" + "0" * 32}))

    spec = importlib.util.spec_from_file_location("convert_k6", KERNEL)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    found = kernel.find_snapshot(tmp_path / "input")
    assert found == snapshot
    reports = kernel.convert_checkpoint(found, tmp_path / "out", kernel.CONVERT, revision=kernel.infer_revision(found))
    card = kernel.model_card(reports, kernel.infer_revision(found))
    assert "4c487e8b" in card
    assert "license: mit" in card
    for component in kernel.CONVERT:
        assert (tmp_path / "out" / component / "model.safetensors").is_file()
