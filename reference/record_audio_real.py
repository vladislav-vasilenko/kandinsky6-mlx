"""Record the float32 reference audio decode (latents -> mel -> waveform) on real weights for spec 07 stage B.

The reference `MMAudioVAE.decode` and `MMAudioVocoder` (diffusers 0.42.1, CPU, float32) are loaded from the converted
directories (`<weights>/audio_vae`, `<weights>/vocoder`) with the convolution weights moved back to the PyTorch
layout. The converter only permutes axes and drops the duplicate vocoder inside `audio_vae` (spec 02 checks values
bit-exact), so this runs the reference on the released tensors without a second download of the diffusers files.

Latents follow the pipeline: DiT-layout audio latents `(B, L, 40)` are permuted to `(B, 40, L)` and divided by
`scaling_factor` before `decode`. With `--latents` (a fixture with `inputs.audio_latents` from a Smoke run) those are
used; otherwise `np.random.default_rng(seed)` normal latents of length `--length` stand in until spec 09 records real
ones.

Usage:
    uv run --group reference python reference/record_audio_real.py --weights weights/lite-distill \\
        --out fixtures/real/audio_smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.utils.fixtures import save_fixture


def load_reference(directory: Path, component: str, cls) -> torch.nn.Module:
    from safetensors.torch import load_file

    model = cls.from_config(json.loads((directory / "config.json").read_text())).float().eval()
    state = {}
    for name, value in load_file(str(directory / "model.safetensors")).items():
        rule = conv_rule(component, name)
        state[name] = value.permute(*np.argsort(rule.axes())) if rule is not None else value
    model.load_state_dict(state, strict=True)
    return model


def main(argv: list[str] | None = None) -> int:
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    from reference.recorder import library_versions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("weights/lite-distill"))
    parser.add_argument("--latents", type=Path, help="fixture with inputs.audio_latents (B, L, C) from a Smoke run")
    parser.add_argument("--length", type=int, default=45, help="latent length when --latents is not given")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("fixtures/real/audio_smoke"))
    args = parser.parse_args(argv)

    audio_vae = load_reference(args.weights / "audio_vae", "audio_vae", MMAudioVAE)
    vocoder = load_reference(args.weights / "vocoder", "vocoder", MMAudioVocoder)
    if args.latents is not None:
        from safetensors.torch import load_file

        latents = load_file(str(args.latents.with_suffix(".safetensors")))["inputs.audio_latents"].float()
        source = str(args.latents)
    else:
        rng = np.random.default_rng(args.seed)
        latents = torch.from_numpy(
            rng.standard_normal((1, args.length, audio_vae.config.latent_channels)).astype(np.float32)
        )
        source = f"np.random.default_rng({args.seed}).standard_normal"
    with torch.no_grad():
        z = latents.permute(0, 2, 1) / audio_vae.config.scaling_factor
        mel = audio_vae.decode(z, return_dict=False)[0]
        waveform = vocoder(mel, return_dict=False)[0]
    save_fixture(
        args.out,
        {"inputs.audio_latents": latents, "outputs.mel": mel, "outputs.waveform": waveform},
        {
            "fixture": args.out.name,
            "reference": "diffusers.MMAudioVAE.decode + MMAudioVocoder.forward",
            "weights": str(args.weights),
            "latents": source,
            "scaling_factor": audio_vae.config.scaling_factor,
            "sample_rate": audio_vae.config.sample_rate,
            "versions": library_versions(),
        },
    )
    print(f"mel {tuple(mel.shape)}, waveform {tuple(waveform.shape)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
